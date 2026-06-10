import gradio as gr
from Retrieval_pipeline import ask_question

def respond(question):
    """Simple function that takes a question and returns answer with confidence"""
    if not question:
        return "Please enter a question."
    
    try:
        answer, sources, confidence = ask_question(question)
        
        # Build response with confidence
        response = f"{confidence['icon']} **Confidence: {confidence['level'].upper()}** ({confidence['score']})\n"
        response += f"{confidence['message']}\n\n"
        response += f"---\n\n"
        response += f"**Answer:** {answer}\n\n"
        response += f"**Sources:** {', '.join(sources)}"
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"









# Create simple interface
demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="Your Question", placeholder="What is machine learning?", lines=2),
    outputs=gr.Markdown(label="Answer"),
    title="RAG Document Q&A",
    description="Ask questions about your documents. Answers include source citations.",
    examples=[
        ["What is machine learning?"],
        ["What are the types of machine learning?"],
        ["What is NLP?"]
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
