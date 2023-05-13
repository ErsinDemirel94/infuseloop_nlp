import os
import openai

openai.api_key = os.getenv("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")


class custom_llm:
    
    def __init__(self):
        self.response = openai.Completion.create(
          model="text-davinci-003",
          prompt="""INPUT\n\"In my previous job, I had a co-worker who was constantly negative and critical towards me and other colleagues. At first, I tried to ignore it and keep a professional attitude, but it was affecting my work and my morale. So, I decided to have a one-on-one conversation with my co-worker and express how their behavior was making me feel.\n\nDuring the conversation, I used \"I\" statements and stayed calm and respectful. I listened to their perspective and acknowledged their concerns, but also made it clear that their behavior was not acceptable. We were able to come to a mutual understanding and they improved their behavior towards me and the rest of the team.\n\nOverall, I learned that it's important to address difficult situations with co-workers in a professional and respectful manner, while also being honest about how their behavior is impacting you and the team.\"\n\nCALCULATION\nphrases:\nSituation: \"had a co-worker who was constantly negative and critical towards me and other colleagues.\"\nTask: \"have a one-on-one conversation with my co-worker\"\nAction:\"I used \"I\" statements and stayed calm and respectful. I listened to their perspective and acknowledged their concerns, but also made it clear that their behavior was not acceptable.\"\nResult:\" We were able to come to a mutual understanding and they improved their behavior towards me and the rest of the team.\"\n\npoints:\n\"had a co-worker who was constantly negative and critical towards me and other colleagues.\" --> clearly explained --> 0.80,\n\"have a one-on-one conversation with my co-worker.\" --> not very clear explanation of the task --> 0.33\n\"I used \"I\" statements and stayed calm and respectful. I listened to their perspective and acknowledged their concerns, but also made it clear that their behavior was not acceptable.\" --> clearly explained, effective action --> 0.92\n\"We were able to come to a mutual understanding and they improved their behavior towards me and the rest of the team.\" --> good, expected result --> 0.94\n\noverall_point\n(0.80+0.33 +0.92+ 0.94) / 4\n\nOUTPUT:\n{\"situation\":\"had a co-worker who was constantly negative and critical towards me and other colleagues.\",\n\"task:\"have a one-on-one conversation with my co-worker\",\n\"action: \"I used \"I\" statements and stayed calm and respectful. I listened to their perspective and acknowledged their concerns, but also made it clear that their behavior was not acceptable.\",\n\"result\":\"We were able to come to a mutual understanding and they improved their behavior towards me and the rest of the team.\",\n\"points\":[\n{\"clearly explained\" : 0.80},\n{\"not very clear explanation of the task\": 0.33},\n{\"clearly explained, effective action\":0.92},\n{\"good, expected result\":0.94}\n],\n\"overall_point\": 0.7475\n}\n\n\"I once worked with a co-worker who was always missing deadlines and not delivering on their responsibilities. It was affecting our team's performance and creating a stressful work environment. I decided to approach the situation by first having a one-on-one conversation with my co-worker to understand if there were any underlying issues that were causing them to miss deadlines.\n\nDuring the conversation, I discovered that my co-worker was overwhelmed with their workload and was struggling to prioritize their tasks effectively. I offered to help by breaking down their responsibilities into manageable tasks and providing support where needed. We also agreed to set clear deadlines and check-in regularly to ensure that they were on track.\n\nBy working together and finding a solution that worked for both of us, we were able to improve our team's performance and create a more positive work environment. This experience taught me the importance of communication, collaboration, and empathy when dealing with difficult co-workers.\"\nOUTPUT:\n{\"situation\":\"worked with a co-worker who was always missing deadlines and not delivering on their responsibilities.\",\n\"task:\"have a one-on-one conversation with my co-worker\",\n\"action:\"I discovered that my co-worker was overwhelmed with their workload and was struggling to prioritize their tasks effectively. I offered to help by breaking down their responsibilities into manageable tasks and providing support where needed. We also agreed to set clear deadlines and check-in regularly to ensure that they were on track.\",\n\"result\":\"By working together and finding a solution that worked for both of us, we were able to improve our team's performance and create a more positive work environment.\",\n\"points\":[\n{\"clearly explained\" : 0.80},\n{\"not very clear explanation of the task\": 0.33},\n{\"clearly explained, effective action\":0.92},\n{\"good, expected result\":0.94}\n],\n\"overall_point\": 0.8425\n}""",
          temperature=0.7,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        
    def query(self,input):
        
        return self.response
        
        template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Respond in Spanish.

QUESTION: {input}
=========
{summaries}
=========
FINAL ANSWER IN SPANISH:"""

# create a prompt template
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

# query 
chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
query = "What did the president say about Justice Breyer?"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
