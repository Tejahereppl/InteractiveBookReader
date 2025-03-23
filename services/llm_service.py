import logging
from typing import List, Dict, Any, Optional
from uuid import UUID


from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document

from config.settings import settings_obj
from models.story import Story, Character, StoryScenario, ChatMessage, ChatSession
from services.vector_db import VectorDBService

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with language models"""
    
    def __init__(self, vector_db_service: VectorDBService):
        self.vector_db = vector_db_service
        self.llm = ChatOpenAI(
            model_name=settings_obj.LLM_MODEL, 
            temperature=settings_obj.TEMPERATURE,
            openai_api_key=settings_obj.OPENAI_API_KEY,
            max_tokens=settings_obj.MAX_TOKENS
        )
    
    async def analyze_story_content(self, story: Story, documents: List[Document]) -> Dict[str, Any]:
        """Analyze story content to extract characters, settings, and plot"""
        try:
            # Combine some of the documents for analysis
            # We don't want to exceed token limits, so we'll use a subset
            text_sample = "\n\n".join([doc.page_content for doc in documents[:10]])
            
            # Create a prompt for story analysis
            story_analysis_prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                You are a literary analyst. Analyze the following text from a book and extract:
                1. Main characters (name, brief description, key traits)
                2. Setting (time period, location, environment)
                3. Main plot themes
                
                Be concise but thorough in your analysis.
                
                TEXT:
                {text}
                
                ANALYSIS:
                """
            )
            
            # Create and run the chain
            analysis_chain = LLMChain(
                llm=self.llm,
                prompt=story_analysis_prompt,
                verbose=True
            )
            
            analysis_result = await analysis_chain.arun(text=text_sample)
            
            # Parse the results (this is simplified - you might want more structured parsing)
            analysis = {
                "raw_analysis": analysis_result,
                "characters": [],
                "setting": "",
                "plot": ""
            }
            
            # Now create a more structured extraction of characters
            character_extraction_prompt = PromptTemplate(
                input_variables=["text", "analysis"],
                template="""
                Based on the following text from a book and your initial analysis, extract a list of main characters.
                For each character, provide:
                - Name
                - Description
                - Key personality traits (comma-separated)
                - Relationships to other characters (if applicable)
                
                Format your response as JSON. Example:
                [
                  {
                    "name": "Character Name",
                    "description": "Brief description",
                    "traits": ["trait1", "trait2", "trait3"],
                    "relationships": {"Another Character": "relationship type"}
                  }
                ]
                
                TEXT SAMPLE:
                {text}
                
                YOUR INITIAL ANALYSIS:
                {analysis}
                
                CHARACTER JSON:
                """
            )
            
            character_chain = LLMChain(
                llm=self.llm,
                prompt=character_extraction_prompt,
                verbose=True
            )
            
            characters_json = await character_chain.arun(text=text_sample, analysis=analysis_result)
            
            # You would parse the JSON here into Character objects
            # This is simplified for now
            import json
            try:
                characters_data = json.loads(characters_json)
                characters = []
                for char_data in characters_data:
                    character = Character(
                        name=char_data.get("name", "Unknown"),
                        description=char_data.get("description", ""),
                        traits=char_data.get("traits", []),
                        relationships=char_data.get("relationships", {})
                    )
                    characters.append(character)
                
                analysis["characters"] = characters
            except json.JSONDecodeError:
                logger.error(f"Failed to parse character JSON: {characters_json}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing story content: {e}")
            raise
    
    async def generate_story_scenarios(self, story: Story, characters: List[Character]) -> List[StoryScenario]:
        """Generate interactive scenarios based on the story content"""
        try:
            # First, get some relevant document chunks
            sample_query = f"What are the main events in {story.metadata.title}?"
            relevant_docs = await self.vector_db.similarity_search(story.id, sample_query, k=10)
            
            # Combine text for context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create a prompt for scenario generation
            scenario_prompt = PromptTemplate(
                input_variables=["title", "context", "characters"],
                template="""
                Create 3 interactive storytelling scenarios based on the book "{title}".
                
                CONTEXT FROM THE BOOK:
                {context}
                
                CHARACTERS:
                {characters}
                
                For each scenario, provide:
                1. A title
                2. A brief description of the scenario setup
                3. Character-specific prompts for the main characters involved
                4. A system prompt that would guide an AI in roleplaying this scenario
                
                Format your response as JSON. Example:
                [
                  {{
                    "title": "Scenario Title",
                    "description": "Scenario description",
                    "character_prompts": {{
                      "Character Name": "Character-specific prompt"
                    }},
                    "system_prompt": "System prompt for the AI"
                  }}
                ]
                
                SCENARIOS:
                """
            )
            
            # Format characters for the prompt
            characters_text = "\n".join([
                f"- {char.name}: {char.description}" for char in characters
            ])
            
            # Create and run the chain
            scenario_chain = LLMChain(
                llm=self.llm,
                prompt=scenario_prompt,
                verbose=True
            )
            
            scenarios_json = await scenario_chain.arun(
                title=story.metadata.title,
                context=context,
                characters=characters_text
            )
            
            # Parse the JSON result
            import json
            try:
                scenarios_data = json.loads(scenarios_json)
                scenarios = []
                
                for scenario_data in scenarios_data:
                    scenario = StoryScenario(
                        story_id=story.id,
                        title=scenario_data.get("title", "Untitled Scenario"),
                        description=scenario_data.get("description", ""),
                        character_prompts=scenario_data.get("character_prompts", {}),
                        system_prompt=scenario_data.get("system_prompt", "")
                    )
                    scenarios.append(scenario)
                
                return scenarios
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse scenarios JSON: {scenarios_json}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating story scenarios: {e}")
            return []
    
    async def chat_as_character(
        self, 
        story: Story, 
        character: Character, 
        user_name: str,
        message: str,
        session: Optional[ChatSession] = None
    ) -> ChatSession:
        """Generate a response as if from a character in the story"""
        try:
            # Create a new session if one doesn't exist
            if session is None:
                session = ChatSession(
                    story_id=story.id,
                    character_id=character.id,
                    user_name=user_name,
                    messages=[]
                )
            
            # Add the user message to the session
            user_message = ChatMessage(
                role="user",
                content=message
            )
            session.messages.append(user_message)
            
            # Get relevant context from the vector database
            relevant_docs = await self.vector_db.similarity_search(
                story.id, 
                message, 
                k=5
            )
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create a chat prompt
            system_template = """
            You are {character_name} from the book "{book_title}".
            
            Character details:
            {character_description}
            
            Key traits: {character_traits}
            
            You should respond as this character would, maintaining their personality, knowledge, and speech patterns.
            The user is roleplaying as {user_name}, a character interacting with you in this story world.
            
            Use the following context from the book to inform your responses, but don't explicitly reference it:
            {context}
            
            Remember to stay in character and create an immersive storytelling experience.
            """
            
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
            
            human_template = "{message}"
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
            
            chat_prompt = ChatPromptTemplate.from_messages([
                system_message_prompt,
                human_message_prompt
            ])
            
            # Format the traits list
            traits_text = ", ".join(character.traits)
            
            # Prepare the conversation history
            # We'll limit it to the last few messages to avoid exceeding token limits
            conversation_history = session.messages[-10:] if len(session.messages) > 10 else session.messages
            conversation_context = "\n".join([
                f"{msg.role}: {msg.content}" for msg in conversation_history[:-1]  # Exclude the latest message which we'll provide separately
            ])
            
            if conversation_context:
                system_template += f"\n\nRecent conversation history:\n{conversation_context}"
            
            # Create the chain
            chat_chain = LLMChain(
                llm=self.llm,
                prompt=chat_prompt,
                verbose=True
            )
            
            # Generate the response
            response = await chat_chain.arun(
                character_name=character.name,
                book_title=story.metadata.title,
                character_description=character.description,
                character_traits=traits_text,
                user_name=user_name,
                context=context,
                message=message
            )
            
            # Add the assistant's response to the session
            assistant_message = ChatMessage(
                role="assistant",
                content=response
            )
            session.messages.append(assistant_message)
            
            # Update the session timestamp
            from datetime import datetime
            session.updated_at = datetime.now()
            
            return session
            
        except Exception as e:
            logger.error(f"Error generating character response: {e}")
            raise