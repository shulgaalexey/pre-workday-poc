"""
Test script to verify all memory types work correctly in create_react_agent.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

def test_all_memory_types():
    """Test that create_react_agent works with all memory configurations."""
    print("Testing all memory types in create_react_agent...")

    # Mock environment to avoid API calls
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        with patch('src.agent.ChatOpenAI') as mock_openai:
            with patch('src.agent.initialize_agent') as mock_init_agent:

                mock_llm = MagicMock()
                mock_openai.return_value = mock_llm
                mock_agent = MagicMock()
                mock_init_agent.return_value = mock_agent

                from src.agent import create_react_agent

                # Test tools
                tools = [MagicMock()]

                # Test 1: In-memory (default)
                print("1. Testing in-memory memory...")
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    f.write('memory: "in-memory"\n')
                    config_path = f.name

                with patch('src.agent.get_project_root') as mock_root:
                    mock_root.return_value.joinpath.return_value = config_path

                    agent = create_react_agent(mock_llm, tools)
                    assert agent is mock_agent
                    print("   ✓ In-memory memory works")

                os.unlink(config_path)

                # Test 2: Persistent SQLite
                print("2. Testing persistent SQLite memory...")
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    f.write('memory: "persistent-sqlite"\n')
                    config_path = f.name

                with patch('src.agent.get_project_root') as mock_root:
                    mock_root.return_value.joinpath.return_value = config_path

                    agent = create_react_agent(mock_llm, tools)
                    assert agent is mock_agent
                    print("   ✓ Persistent SQLite memory works")

                os.unlink(config_path)

                # Test 3: Vector store
                print("3. Testing vector store memory...")
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    f.write('memory: "vector-store"\n')
                    config_path = f.name

                with patch('src.agent.get_project_root') as mock_root:
                    with patch('src.agent._get_vector_memory') as mock_vector_mem:
                        mock_vector_memory = MagicMock()
                        mock_vector_mem.return_value = mock_vector_memory

                        mock_root.return_value.joinpath.return_value = config_path

                        agent = create_react_agent(mock_llm, tools)
                        assert agent is mock_agent
                        mock_vector_mem.assert_called_once()
                        print("   ✓ Vector store memory works")

                os.unlink(config_path)

                print("\n✅ All memory types tested successfully!")

if __name__ == "__main__":
    test_all_memory_types()
