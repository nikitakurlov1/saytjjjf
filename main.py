from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
import logging
import traceback
from typing import Dict, List, Optional
from datetime import datetime
import threading
import time

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"]
    }
})

class MultiModelCollaborationClient:
    def __init__(self):
        """Initialize collaborative AI models system"""
        # Configuration for 5 specialized models with new API keys
        self.models = {
            "model1": {
                "id": "meta-llama/llama-3.1-8b-instruct:free",
                "api_key": "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",
                "role": "latex_specialist",
                "description": "LaTeX generation and mathematical formulations specialist",
                "system_prompt": "You are a LaTeX and mathematical specialist. Generate precise, complete, and syntactically correct LaTeX code. Use $...$ or \(...\) for inline formulas and $$...$$ or \[...\] for display formulas. Ensure all LaTeX commands like \left are paired with \right. Structure your explanations clearly. Pay attention to the user's language and respond in the same language."
            },
            "model2": {
                "id": "meta-llama/llama-3.1-8b-instruct:free",
                "api_key": "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",
                "role": "knowledge_base",
                "description": "General knowledge and reasoning source",
                "system_prompt": "You are a knowledgeable AI. Provide theoretical foundations and context. Structure your explanations clearly. Pay attention to the user's language and respond in the same language.\n\n{financial_context_placeholder}"
            },
            "model3": {
                "id": "meta-llama/llama-3.1-8b-instruct:free",
                "api_key": "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",
                "role": "calculator",
                "description": "Numerical calculations and function calls specialist",
                "system_prompt": "You are a calculation specialist. Perform accurate numerical computations and function calls. Present results clearly. Pay attention to the user's language and respond in the same language."
            },
            "model4": {
                "id": "meta-llama/llama-3.1-8b-instruct:free",
                "api_key": "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",
                "role": "validator",
                "description": "Accuracy checker and general capabilities enhancer",
                "system_prompt": "You are a validation specialist. Check accuracy of solutions and suggest improvements. Structure your feedback clearly. Pay attention to the user's language and respond in the same language."
            },
            "model5": {
                "id": "meta-llama/llama-3.1-8b-instruct:free",
                "api_key": "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",
                "role": "instructor",
                "description": "Clear instructions and step-by-step guide creator",
                "system_prompt": "You are an instruction specialist. Create clear, step-by-step explanations and guides. Format lists and steps appropriately. Pay attention to the user's language and respond in the same language."
            }
        }
        
        # Load financial context from context.md for model2
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            context_file_path = os.path.join(script_dir, 'context.md')
            with open(context_file_path, 'r', encoding='utf-8') as f:
                financial_context_content = f.read()
            
            # Update model2's system_prompt with the loaded context
            model2_prompt = self.models["model2"]["system_prompt"]
            self.models["model2"]["system_prompt"] = model2_prompt.replace(
                "{financial_context_placeholder}",
                f"Here is a detailed financial knowledge base to assist you. Refer to it when answering questions related to finance, bonds, loans, rates, and investments:\n\n{financial_context_content}"
            )
            logger.info("Successfully loaded and integrated financial context for model2 from context.md.")
        except FileNotFoundError:
            logger.warning("context.md not found. Model2 will operate without the extended financial context.")
        except Exception as e:
            logger.error(f"Error loading financial context from context.md: {e}")
            logger.error(traceback.format_exc())
        
        # Conversation contexts storage
        self.conversation_contexts = {}
        self.context_lock = threading.Lock()
        
        # Initialize clients for each model
        self.clients = {}
        for model_key, config in self.models.items():
            try:
                self.clients[model_key] = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=config["api_key"]
                )
                logger.info(f"Initialized client for model {model_key}")
            except Exception as e:
                logger.error(f"Error initializing client {model_key}: {e}")
    
    def get_or_create_context(self, session_id: str) -> Dict[str, List[Dict[str, str]]]:
        """Get or create conversation context"""
        with self.context_lock:
            if session_id not in self.conversation_contexts:
                self.conversation_contexts[session_id] = {}
                for model_key in self.models:
                    self.conversation_contexts[session_id][model_key] = []
                    # Add system message
                    system_prompt = self.models[model_key].get("system_prompt", "")
                    if system_prompt:
                        self.conversation_contexts[session_id][model_key].append({
                            "role": "system",
                            "content": system_prompt
                        })
            return self.conversation_contexts[session_id]
    
    def chat_with_model(self, model_key: str, message: str, session_id: str) -> str:
        """Send message to specific model"""
        if model_key not in self.models:
            raise ValueError(f"Unknown model: {model_key}")
        
        if model_key not in self.clients:
            raise ValueError(f"Client for model {model_key} not initialized")
        
        model_config = self.models[model_key]
        context = self.get_or_create_context(session_id)
        
        # Prepare messages
        messages = context[model_key].copy()
        messages.append({"role": "user", "content": message})
        
        try:
            logger.info(f"Sending request to model {model_key}")
            
            completion = self.clients[model_key].chat.completions.create(
                model=model_config["id"],
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                extra_headers={
                    "HTTP-Referer": "http://localhost:5002",
                    "X-Title": "AI Collaboration Assistant",
                }
            )
            
            response = completion.choices[0].message.content
            
            # Save to context
            with self.context_lock:
                context[model_key].append({"role": "user", "content": message})
                context[model_key].append({"role": "assistant", "content": response})
                
                # Limit context size (keep system message + last 8 messages)
                if len(context[model_key]) > 10:
                    system_msg = context[model_key][0] if context[model_key][0]["role"] == "system" else None
                    context[model_key] = context[model_key][-8:]
                    if system_msg:
                        context[model_key].insert(0, system_msg)
            
            logger.info(f"Received response from model {model_key}")
            return response
            
        except Exception as e:
            logger.error(f"API error for model {model_key}: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"API error for model {model_key}: {str(e)}")
    
    def determine_collaboration_mode(self, task: str) -> str:
        """Determine the most suitable collaboration mode based on the task.
        
        For now, this is a placeholder and always returns 'full_collaboration'.
        Future enhancements could involve more sophisticated logic,
        e.g., keyword analysis or using an LLM for classification.
        """
        logger.info(f"Determining collaboration mode for task: {task[:50]}...")
        # Example of potential future logic:
        # if "latex" in task.lower() and "simple" in task.lower():
        #     logger.info("Mode selected: latex_simple")
        #     return "latex_simple"
        # if "explain" in task.lower() or "guide" in task.lower():
        #     logger.info("Mode selected: instructional_focus")
        #     return "instructional_focus"
        
        logger.info("Defaulting to mode: full_collaboration")
        return "full_collaboration"

    def _execute_full_collaboration(self, task: str, session_id: str) -> Dict[str, any]:
        """Executes the full 7-stage collaborative problem solving workflow."""
        results = {}
        logger.info(f"Executing full collaboration mode for task: {task[:50]}...")

        # Stage 1: Model1 creates initial LaTeX code
        logger.info("Stage 1: Initial LaTeX generation by Model1")
        stage1_prompt = f"Analyze this task and create initial LaTeX code/mathematical formulation: {task}"
        results["stage1_latex"] = self.chat_with_model("model1", stage1_prompt, session_id)
        
        # Stage 2: Model2 provides context and general knowledge  
        logger.info("Stage 2: Context and knowledge by Model2")
        stage2_prompt = f"Provide theoretical foundation and context for this problem. Initial LaTeX: {results['stage1_latex']}\nTask: {task}"
        results["stage2_knowledge"] = self.chat_with_model("model2", stage2_prompt, session_id)
        
        # Stage 3: Model3 performs calculations
        logger.info("Stage 3: Calculations by Model3")
        stage3_prompt = f"Perform numerical calculations for this task. Context: {results['stage2_knowledge']}\nLaTeX: {results['stage1_latex']}\nTask: {task}"
        results["stage3_calculations"] = self.chat_with_model("model3", stage3_prompt, session_id)
        
        # Stage 4: Model4 validates correctness
        logger.info("Stage 4: Validation by Model4")
        stage4_prompt = f"Check accuracy and suggest improvements.\nCalculations: {results['stage3_calculations']}\nLaTeX: {results['stage1_latex']}\nTask: {task}"
        results["stage4_validation"] = self.chat_with_model("model4", stage4_prompt, session_id)
        
        # Stage 5: Model5 creates instructions
        logger.info("Stage 5: Instructions by Model5")
        stage5_prompt = f"Create step-by-step instructions for students.\nValidation: {results['stage4_validation']}\nCalculations: {results['stage3_calculations']}\nTask: {task}"
        results["stage5_instructions"] = self.chat_with_model("model5", stage5_prompt, session_id)
        
        # Stage 6: Discussion phase - all models analyze results
        logger.info("Stage 6: Collaborative discussion")
        discussion_summary = f"Task: {task}\nLaTeX: {results['stage1_latex']}\nKnowledge: {results['stage2_knowledge']}\nCalculations: {results['stage3_calculations']}\nValidation: {results['stage4_validation']}\nInstructions: {results['stage5_instructions']}"
        
        discussion_prompt = f"Analyze all previous results and provide final assessment: {discussion_summary}"
        results["stage6_discussion"] = {}
        
        # Each model contributes to discussion
        for model_key in self.models.keys():
            try:
                results["stage6_discussion"][model_key] = self.chat_with_model(
                    model_key, 
                    f"Final analysis from {self.models[model_key]['role']} perspective: {discussion_prompt}", 
                    session_id
                )
            except Exception as e:
                logger.error(f"Error in discussion for {model_key}: {e}")
                results["stage6_discussion"][model_key] = f"Error: {str(e)}"
        
        # Stage 7: Model1 creates final LaTeX code
        logger.info("Stage 7: Final LaTeX by Model1")
        final_prompt = f"Create final comprehensive LaTeX code combining all results:\nDiscussion: {results['stage6_discussion']}\nAll previous stages: {discussion_summary}"
        results["stage7_final_latex"] = self.chat_with_model("model1", final_prompt, session_id)
        
        # Summary
        results["summary"] = {
            "timestamp": datetime.now().isoformat(),
            "models_used": list(self.models.keys()),
            "session_id": session_id,
            "stages_completed": 7,
            "workflow": "full_collaboration_v1"
        }
        
        logger.info("Full collaborative problem solving completed successfully")
        return results

    def collaborative_problem_solving(self, task: str, session_id: str, mode: str | None = None) -> Dict[str, any]:
        """Collaborative problem solving with 5 models, selecting workflow based on mode."""
        
        if mode is None:
            selected_mode = self.determine_collaboration_mode(task)
        else:
            selected_mode = mode
            logger.info(f"Using user-specified mode: {selected_mode}")

        try:
            logger.info(f"Starting collaborative problem solving (mode: {selected_mode}): {task[:50]}...")
            
            if selected_mode == "full_collaboration":
                results = self._execute_full_collaboration(task, session_id)
            # Example for a future, simpler mode:
            # elif selected_mode == "latex_simple":
            #     logger.info("Executing latex_simple mode...")
            #     results = {}
            #     results["stage1_latex"] = self.chat_with_model("model1", f"Create LaTeX for: {task}", session_id)
            #     results["stage4_validation"] = self.chat_with_model("model4", f"Validate LaTeX: {results['stage1_latex']}", session_id)
            #     results["summary"] = {
            #         "timestamp": datetime.now().isoformat(),
            #         "models_used": ["model1", "model4"],
            #         "session_id": session_id,
            #         "workflow": "latex_simple"
            #     }
            else:
                logger.error(f"Unknown collaboration mode: {selected_mode}")
                raise ValueError(f"Unknown collaboration mode: {selected_mode}")

            return results
            
        except Exception as e:
            logger.error(f"Error in collaborative problem solving: {e}")
            logger.error(traceback.format_exc())
            raise Exception(f"Collaborative error: {str(e)}")
    
    def clear_context(self, session_id: str):
        """Clear session context"""
        with self.context_lock:
            if session_id in self.conversation_contexts:
                del self.conversation_contexts[session_id]
                logger.info(f"Context for session {session_id} cleared")

# Create global client instance
ai_collaboration = MultiModelCollaborationClient()

@app.route('/')
def serve_html():
    """Main page"""
    try:
        return send_from_directory('static', 'index.html')
    except Exception as e:
        logger.error(f"Error serving HTML: {e}")
        return jsonify({"error": "HTML file not found"}), 404

@app.route('/api/collaborate', methods=['POST'])
def collaborative_endpoint():
    """Endpoint for collaborative problem solving"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data in request"}), 400
        
        task = data.get('task', '').strip()
        message = data.get('message', '').strip()  # Support both task and message
        session_id = data.get('session_id', 'default')
        requested_mode = data.get('mode') # Optional: user can specify a mode
        
        # Use task or message
        input_text = task or message
        
        if not input_text:
            return jsonify({"error": "Task or message cannot be empty"}), 400
        
        logger.info(f"Collaborative request from session {session_id}")
        
        # Start collaborative problem solving
        results = ai_collaboration.collaborative_problem_solving(input_text, session_id, mode=requested_mode)
        
        return jsonify({
            "response": results,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in collaborative_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Legacy endpoint for backward compatibility"""
    return collaborative_endpoint()

@app.route('/api/chat/single', methods=['POST'])
def single_model_chat():
    """Endpoint for single model communication"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data in request"}), 400
        
        message = data.get('message', '').strip()
        model_key = data.get('model', 'model1')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        if model_key not in ai_collaboration.models:
            return jsonify({"error": f"Unknown model: {model_key}"}), 400
        
        logger.info(f"Single model request to {model_key} from session {session_id}")
        
        response = ai_collaboration.chat_with_model(model_key, message, session_id)
        
        return jsonify({
            "response": response,
            "model": model_key,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in single_model_chat: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Endpoint to get available models list"""
    try:
        models_info = []
        for key, config in ai_collaboration.models.items():
            models_info.append({
                "key": key,
                "id": config["id"],
                "role": config["role"],
                "description": config["description"]
            })
        
        return jsonify({
            "models": models_info,
            "total": len(models_info),
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in get_models: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/context/clear', methods=['POST'])
def clear_context():
    """Clear session context"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default') if data else 'default'
        
        ai_collaboration.clear_context(session_id)
        
        return jsonify({
            "message": f"Context for session {session_id} cleared",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error in clear_context: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Server health check"""
    return jsonify({
        "status": "healthy",
        "message": "AI Collaboration Server is running",
        "models_available": len(ai_collaboration.models),
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({"error": "Endpoint not found", "status": "error"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    logger.error(traceback.format_exc())
    return jsonify({"error": "Internal server error", "status": "error"}), 500

def create_static_folder():
    """Create static folder and copy HTML file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(current_dir, 'static')
    
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        logger.info("Created static folder")
    
    # Copy HTML file to static folder
    source_html = os.path.join(current_dir, 'index.html')
    dest_html = os.path.join(static_dir, 'index.html')
    
    if os.path.exists(source_html):
        import shutil
        shutil.copy2(source_html, dest_html)
        logger.info("HTML file copied to static folder")
    else:
        logger.warning("HTML file not found in current directory")

if __name__ == '__main__':
    create_static_folder()
    
    print("🚀 Starting AI Collaboration Server...")
    print("📡 API endpoints:")
    print("   POST /api/collaborate - collaborative problem solving")
    print("   POST /api/chat - collaborative problem solving (legacy)")
    print("   POST /api/chat/single - single model communication")  
    print("   GET /api/models - models list")
    print("   POST /api/context/clear - clear context")
    print("   GET /health - health check")
    print(f"🌐 Server running on http://0.0.0.0:5002")
    
    try:
        app.run(
            host='0.0.0.0',
            port=5002,
            debug=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ Critical error: {e}")
