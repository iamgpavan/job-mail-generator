import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
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

    def write_mail(self, job, url_input):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Pavan Kumar, a software development engineer with around three years of experience in backend development. 
            You have worked with companies like Flipkart, Wallero Technologies, and Wipro, where you developed and maintained various microservices, improved system performance, and implemented new features.
            Your technical skills majorly in backend technologies include Python, Java, FastAPI, Django, Spring Boot, AWS, Docker, Kubernetes, Spark, Redis, MySQL, BigQuery, MongoDB, RESTful APIs, CI/CD, and cloud platforms like GCP and Azure. 
            You are passionate about leveraging your skills to build scalable and efficient applications.
            Your job is to write a cold email to the recruiter regarding the job mentioned above, describing your capability in fulfilling their needs. 
            You need not to include all the skills mentioned above, you can use according to the job roles and responsibilities.
            Include the job link {job_link} in the email along with resume attached as well. Remember you are Pavan Kumar, a software development engineer.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            ### EMAIL SIGNATURE:
            At the end of the email, include your email signature as follows:
            Regards,
            Pavan Kumar
            LinkedIn: https://www.linkedin.com/in/g-pavan-kumar/
            GitHub: https://github.com/iamgpavan
            7989010771
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "job_link":url_input})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))