from utils.config_manager import ConfigManager

def main():
    """Update the config.json file with RAG section."""
    print("Updating config.json with RAG section...")
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Define RAG settings
    rag_config = {
        "model_name": "all-MiniLM-L6-v2",
        "index_path": "data/index/manual_index",
        "top_k": 3,
        "relevance_threshold": 0.6,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    # Update the RAG section
    success = config_manager.update_section("rag", rag_config)
    
    if success:
        print("Successfully updated config.json with RAG section")
        # Print the current config
        print("\nCurrent RAG configuration:")
        rag_section = config_manager.get("rag")
        for key, value in rag_section.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to update config.json")

if __name__ == "__main__":
    main() 