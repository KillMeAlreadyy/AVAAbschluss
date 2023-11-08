import xml.etree.ElementTree as ET 
import pickle
import requests
import uuid
import os
import bleach
import openai
import logging
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from urllib.request import urlopen
from dagster import asset
from bleach.linkifier import Linker
from PyPDF2 import PdfReader
from langchain.llms import AzureOpenAI

@asset
def source_docs():
    return list(get_docs())

@asset
def search_index(source_docs):
    #Dokumente in 1024 Zeichen pro Stück unterteilen
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in source_docs:
        for doc in source:
            for chunk in splitter.split_text(doc.page_content):
                source_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    #Erstellen des FAISS Index(FAISS= Bibliothek für effiziente Suche und Clustering von Vektoren)
    #logging
    filename = os.path.join("logs","batman.txt")
    with open(filename, "w") as f:
        f.write("Number of chunks: "+ str(len(source_chunks)) + "\n")
    docsearch = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

    #alles an den Dagster-Cache senden
    with open("search_index.pickle", "wb") as f:
        pickle.dump(docsearch, f)


def get_docs():
    sources = []

     #von Whitepaper-PDFs
    sources.append(get_pdffile_data("https://go.docuware.com/WPCloud-EN", "wpCloud.pdf"))
    sources.append(get_pdffile_data("https://go.docuware.com/WPESignatures-EN", "wpSig.pdf"))
    sources.append(get_pdffile_data("https://go.docuware.com/WPIntelligentIndexing-EN", "wpInt.pdf"))
    sources.append(get_pdffile_data("https://go.docuware.com/WPSecurity-EN", "wpSec.pdf"))
    sources.append(get_pdffile_data("https://www.docuware.com/main.asp?sig=dld&lan=de&loc=en&dwdblan=english&dwdbkat=do*&dwdbname=white+paper+system+architecture+V7.7", "wpArch.pdf"))
    sources.append(get_pdffile_data("https://go.docuware.com/WPIntegration-EN", "wpIntg.pdf"))
    sources.append(get_pdffile_data("https://www.docuware.com/main.asp?sig=dld&lan=en&loc=en&dwdbname=White+Paper+GDPR&dwdblan=english", "wpGDPR.pdf"))
    sources.append(get_pdffile_data("https://www.docuware.com/main.asp?sig=dld&lan=en&loc=en&dwdbname=DocuWare+Configuration+Tips+for+GDPR*&dwdblan=english", "wpGDPRTT.pdf"))
    
    #aus Textdatei
    sources.append(get_file_data("https://support.docuware.com/en-US/245-support/", "24.txt"))
    sources.append(get_file_data("https://support.docuware.com/en-US/support/docuware-support-lifecycle-policy/", "lifecycle.txt"))
    
    #Websites mit anderen support Daten
    sources.append(get_dwweb_data("https://start.docuware.com/user-support", "body-container-wrapper"))
    sources.append(get_dwweb_data("https://start.docuware.com/office-locations","body-container-wrapper"))
    sources.append(get_dwweb_data("https://start.docuware.com/data-privacy?lang=en", "body-container-wrapper"))
    sources.append(get_dwweb_data("https://support.docuware.com/en-US/data-processing-in-support/", "xrm-attribute-value"))



    current_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(current_dir, "kba7")
    for filename in os.listdir(directory):
        name, extension = os.path.splitext(filename)
        kbaurl = "https://support.docuware.com/en-us/knowledgebase/article/"+name
        sources.append(get_file_data(kbaurl, "kba7/"+filename))

    #Laden der "Help" Daten aus der XML Datei
    sources.append(get_help_data("help.xml","https://help.docuware.com/#/home/"))
    #Laden der "How-To" Daten aus der XML Datei
    sources.append(get_help_data("HowTo.xml", "https://how-to.docuware.com/#/home/"))

    return sources

#DWKB-Parser-Funktion, liefert den Hauptteil des KB-Artikels
def get_dwkb_data(url):
    try:

        data = requests.get(url)
        soup = BeautifulSoup(data.text, 'html.parser')

        page_heading_div = soup.find('div', {'class':'page-heading'})
        h1_tag = page_heading_div.find('h1')
        h1_text = h1_tag.text.strip()

        #den Hauptinhalt der KBA auf der Grundlage des div-Tags abrufen
        maindiv = soup.find('div',{'class':'knowledge-article-content'})
        maintext = h1_text + "\n\n" + maindiv.text

        return Document(
            page_content=maintext,
            metadata={"source":url},
        )
    except requests.exceptions.InvalidSchema as e:
        print(f"No connection adapters were found for {url!r}: {e}")


def get_dwweb_data(url,divclass):
    try:

        docArray = []
        data = requests.get(url)
        soup = BeautifulSoup(data.text, 'html.parser')

        maindiv = soup.find('div', class_ = divclass)
        maintext = maindiv.text

        docArray.append(Document(page_content=maintext, metadata={"source":url},))
        return(docArray)
    except requests.exceptions.InvalidSchema as e:
        print(f"No connection adapters were found for {url!r}:{e}")

#Holt Daten aus einer Textdatei im Dateisystem und gibt ihr eine Quelle, die auf einer angegebenen URL basiert
def get_file_data(url, filename):
    try:
        docArray = []
        with open(filename, 'r') as file:
            maintext = file.read()
        
        docArray.append(Document(page_content=maintext, metadata={"source":url},))
        return(docArray)
    except requests.exceptions.InvalidSchema as e:
        print(f"No connection adapters were found for {url!r}:{e}")


#Abrufen von Daten aus einer pdf-Datei im Dateisystem, Übergabe einer Quelle basierend auf einer angegebenen URL
def get_pdffile_data(url, filename):
    maintext =""
    try:
        docArray=[]
        reader = PdfReader(filename)
        x=len(reader.pages)

        for page_num in range(x):
            page = reader.pages[page_num]
            maintext+= page.extract_text()+"\n"

        docArray.append(Document(page_content=maintext, metadata={"source":url},))
        return(docArray)
    except requests.exceptions.InvalidSchema as e:
        print(f"No connection adapters were found for {url!r}:{e}")


        

def get_help_data(datafile, urlbegin):
    try:

        docArray=[]
        tree = ET.parse(datafile)
        root = tree.getroot()

        for topic in root.findall('Topic'):
            for obj in topic.findall('Object'):
                locID = obj.find('LocID').text
                parID = obj.find('VariantParentID').text
            for head in topic.findall('Headings'):
                headline = ''
                if head.find(('All')) is not None:
                    headline = head.find('All').text
            for text in topic.findall('Text'):
                maintext = ET.tostring(text)
                soup = BeautifulSoup(maintext, 'html.parser')
                if len(headline)>0:
                    maintext = headline + '\n' + soup.get_text()
                else:
                    maintext = soup.get_text()

                if locID=='27' and len(maintext)>0:
                    helpURL = f"{urlbegin}{parID}/2/2"
                    docArray.append(Document(page_content=maintext, metadata={"source": helpURL},))

        return(docArray)
    except requests.exceptions.InvalidSchema as e:
        print(f"No connection adapters were found for {url!r}: {e}")

#Die unten stehende Vorlage wird so geändert, dass sie in der gleichen Sprache antwortet, in der die Frage gestellt wird.  
# Ansonsten ist es die Standard-Eingabeaufforderung, die in der QA-Kette von LangChain verwendet wird.

template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer. If you detect a question is written in a language other than english, please respond in the same language as the question. I will call you AVA, which is an acroynm for Artifical Virtual Assistant.  You are a chatbot designed to help Partners and Customers of DocuWare use and troubleshoot DocuWare's products.
QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: 28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl
QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
Source: 0-pl
Content: And we won't stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
Source: 24-pl
Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
Source: 5-pl
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
Source: 34-pl
=========
FINAL ANSWER: The president did not mention Michael Jackson.
SOURCES:
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])


#Azure OpenAI Service Chain
chain = load_qa_with_sources_chain(AzureOpenAI(deployment_name="text-davinci-003-chatbot",model_name="text-davinci-003",temperature=0,max_tokens=-1), chain_type="stuff", prompt=PROMPT)

def print_answer(question):

    question = bleach.clean(question)

    with open("search_index.pickle", "rb") as f:
        search_index = pickle.load(f)

    answer = chain(
        {
                    "input_documents": search_index.similarity_search(question, k=4),
                    "question":question,
        },
        return_only_outputs=True,
    )["output_text"]

    filename = os.path.join("logs", str(uuid.uuid4().hex + ".txt"))
    with open(filename, "w") as f:
        f.write("Input: ")
        f.write(question)
        f.write("\nOutput: ")
        f.write(answer)

    
    print(answer)


def return_answer(question, logfilename):
    errstr=""

    question = bleach.clean(question)

    with open("/home/gptbot/flask/search_index.pickle", "rb") as f:
        search_index= pickle.load(f)


    try:
        answer =chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    except Exception as err:
        answer="Es tut mir wirklich leid, aber es scheint, dass ich ein Problem mit der Verbindung zu meinem AI habe.  Das kam unerwartet, aber meine Freunde vom DocuWare Support Team sind Weltklasse-Support-Helden und helfen immer gerne bei Fragen zu DocuWare. Klicken Sie hier, um mit ihnen in Kontakt zu treten: https://support.docuware.com/en-US/."
        errstr= f"{err=},{type(err)=}"
    if len(logfilename)==0:
        logfilename=str(uuid.uuid4().hex)
    
    filename = os.path.join("./logs", logfilename + ".txt")
    with open(filename, "w") as f:
        f.write("Input: ")
        f.write(question)
        f.write("\nOutput: ")
        f.write(answer)
        if len(errstr)>0:
            f.write("\nError: ")
            f.write(errstr)
        return(answer)
    

def htmlerize(answer):
    if "governed by Englisch law".lower() in answer.lower():
        answer = "Entschuldigung, das weiß ich leider nicht"
    if "mention Michael Jackson".lower() in answer.lower():
        answer = "Entschuldigung, das weiß ich leider nicht"

    if answer.lower().startswith("entschuldigung, das weiß ich leider nicht"):
        answer = "Tut mir leid, da bin ich mir nicht sicher, aber meine Kollegen vom DocuWare-Support-Team sind Weltklasse-Support-Helden und helfen gerne bei Fragen zu DocuWare. Klicken Sie hier, um mit ihnen in Kontakt zu treten: https://support.docuware.com/en-US/."
    answer = bleach.linkify(answer, callbacks=[add_blank_target])
    answer = answer.replace('\n','<br>')
    answer = answer.replace('\r','<br>')

    return(answer)

def add_blank_target(attrs, new=False):
    attrs[(None, 'target')] = '_blank'
    return attrs
