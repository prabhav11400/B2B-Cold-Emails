import os
import re
import uuid
import streamlit as st
from dotenv import load_dotenv  # Import dotenv to load environment variables
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import chromadb


# Load environment variables from the .env file
load_dotenv()

# Utility Functions
def clean_text(text):
    """Cleans raw text by removing HTML tags, URLs, special characters, and extra whitespace."""
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    text = ' '.join(text.split())
    return text


# Portfolio Management
class Portfolio:
    def __init__(self):
        # Portfolio data is embedded here instead of reading from a file
        self.data = [
            {"Techstack": "React, Node.js, MongoDB", "Links": "https://example.com/react-portfolio"},
            {"Techstack": "Angular,.NET, SQL Server", "Links": "https://example.com/angular-portfolio"},
            {"Techstack": "Vue.js, Ruby on Rails, PostgreSQL", "Links": "https://example.com/vue-portfolio"},
            {"Techstack": "Python, Django, MySQL", "Links": "https://example.com/python-portfolio"},
            {"Techstack": "Java, Spring Boot, Oracle", "Links": "https://example.com/java-portfolio"},
            {"Techstack": "Flutter, Firebase, GraphQL", "Links": "https://example.com/flutter-portfolio"},
            {"Techstack": "WordPress, PHP, MySQL", "Links": "https://example.com/wordpress-portfolio"},
            {"Techstack": "Magento, PHP, MySQL", "Links": "https://example.com/magento-portfolio"},
            {"Techstack": "React Native, Node.js, MongoDB", "Links": "https://example.com/react-native-portfolio"},
            {"Techstack": "iOS, Swift, Core Data", "Links": "https://example.com/ios-portfolio"},
            {"Techstack": "Android, Java, Room Persistence", "Links": "https://example.com/android-portfolio"},
            {"Techstack": "Kotlin, Android, Firebase", "Links": "https://example.com/kotlin-android-portfolio"},
            {"Techstack": "Android TV, Kotlin, Android NDK", "Links": "https://example.com/android-tv-portfolio"},
            {"Techstack": "iOS, Swift, ARKit", "Links": "https://example.com/ios-ar-portfolio"},
            {"Techstack": "Cross-platform, Xamarin, Azure", "Links": "https://example.com/xamarin-portfolio"},
            {"Techstack": "Backend, Kotlin, Spring Boot", "Links": "https://example.com/kotlin-backend-portfolio"},
            {"Techstack": "Frontend, TypeScript, Angular", "Links": "https://example.com/typescript-frontend-portfolio"},
            {"Techstack": "Full-stack, JavaScript, Express.js", "Links": "https://example.com/full-stack-js-portfolio"},
            {"Techstack": "Machine Learning, Python, TensorFlow", "Links": "https://example.com/ml-python-portfolio"},
            {"Techstack": "DevOps, Jenkins, Docker", "Links": "https://example.com/devops-portfolio"},
        ]
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """Loads the portfolio into the ChromaDB collection."""
        if not self.collection.count():
            for row in self.data:
                self.collection.add(
                    documents=row["Techstack"],
                    metadatas={"links": row["Links"]},
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):
        """Queries the collection for relevant links based on the skills provided."""
        return self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])


# Chain Logic
class Chain:
    def __init__(self):
        # Fetch the API key securely from environment variables
        groq_api_key = os.getenv('GROQ_API_KEY')
        print("GROQ_API_KEY:", groq_api_key)  # Debugging line to check if the key is loaded

        if not groq_api_key:
            raise ValueError("API key is missing. Ensure it is set in the .env file or as an environment variable.")
        
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,  # Use the API key from the environment
            model_name="llama-3.1-70b-versatile"
        )


    def extract_jobs(self, cleaned_text):
        """Extracts job descriptions from scraped text."""
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        """Generates a cold email based on job details and portfolio links."""
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            Your name is Aditya Singh, "Prabhav Singh ka bhai". Contact : adityasingh11400@gmail.com. You are an executive at **Prabhav_Streamlit_App Consultancy**, a cutting-edge consultancy providing tech solutions 
            tailored to diverse business needs. Our team of experts has successfully executed numerous projects in the following areas: 
            automation, process optimization, and tailored software solutions. 

            Your job is to write a personalized email to the client regarding their job posting, showcasing **Prabhav_Streamlit_App Consultancy**'s 
            capability to meet their needs by matching the following relevant portfolio examples: {link_list}.
            
            Please highlight why we are the best match for their job posting and how our employees can fulfill their requirements. 
            Do not add a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content


# Streamlit App
def create_streamlit_app(llm, portfolio):
    """Creates the Streamlit interface for the Cold Mail Generator."""
    st.title("📧 Cold Mail Generator - Prabhav_Streamlit_App Consultancy")
    url_input = st.text_input("Enter a Job Posting URL:", value="Enter the link of Job Description to know the relevant portfolios from out company ... ")
    submit_button = st.button("Generate Email")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:  # for every job on the webpage, I am doing this
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


# Main Function
if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio)
