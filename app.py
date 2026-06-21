import gradio as gr
from Retrieval_pipeline import RAG_Pipeline
from Ingestion_pipeline import all_chunks_with_metadata

# Initialize the RAG pipeline once
rag = RAG_Pipeline()


if all_chunks_with_metadata:
    rag.load_chunks_for_bm25(all_chunks_with_metadata)
    print("✅ BM25 index loaded")
else:
    print("⚠️ No chunks found - run Ingestion_pipeline.py first")


def respond(question):
    if not question:
        return "Please enter a question."
    
    try:
        answer, sources, confidence = rag.ask_question(question)
        
        if confidence:
            response = f"{confidence['icon']} **Confidence: {confidence['level'].upper()}** ({confidence['score']})\n"
            response += f"{confidence['message']}\n\n"
            response += f"---\n\n"
            response += f"**Answer:** {answer}\n\n"
            response += f"**Sources:** {', '.join(sources)}"
        else:
            response = f"**Answer:** {answer}\n\n"
            response += f"**Sources:** {', '.join(sources)}"
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="Your Question", placeholder="What is machine learning?", lines=2),
    outputs=gr.Markdown(label="Answer"),
    title="RAG Document Q&A with Hybrid Search",
    description="Ask questions about your documents. Uses hybrid search (vector + BM25) with RRF fusion.",
    examples=[
        ["What is machine learning?"],
        ["What are the types of machine learning?"],
        ["What is NLP?"]
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)