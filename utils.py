import base64
from IPython.display import HTML, display
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod


def draw_mermaid_png(agent, height="70vh", width="min(90vw, 300px)"):
    """Render a Mermaid graph using the Mermaid.ink API for notebook display.

    The diagram keeps its aspect ratio, shrinks if it would exceed the viewport
    height, and otherwise stays readable without manual tuning.
    """
    png_bytes = agent.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    b64_img = base64.b64encode(png_bytes).decode()
    
    # save to file for debugging
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
        
    return png_bytes
