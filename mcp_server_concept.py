#!/usr/bin/env python3
"""
MCP Server Concept für YouTube Summarizer
Ermöglicht anderen AI-Tools Zugriff auf unsere Video-Sammlung
"""

import json
from typing import Dict, List, Any

class YouTubeSummarizerMCP:
    """MCP Server für YouTube Video-Sammlung"""
    
    def __init__(self):
        self.tools = {
            "search_videos": {
                "name": "search_videos",
                "description": "Search through YouTube video summaries using semantic search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for finding relevant videos"
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            },
            "get_video_details": {
                "name": "get_video_details", 
                "description": "Get detailed information about a specific video",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "YouTube video ID"
                        }
                    },
                    "required": ["video_id"]
                }
            },
            "ask_video_question": {
                "name": "ask_video_question",
                "description": "Ask a question about the video collection and get AI-powered answers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question about the video content"
                        }
                    },
                    "required": ["question"]
                }
            }
        }
        
        self.resources = {
            "video_collection": {
                "uri": "youtube://collection/all",
                "name": "Complete Video Collection",
                "description": "Access to all YouTube video summaries",
                "mimeType": "application/json"
            }
        }
    
    def get_server_info(self) -> Dict[str, Any]:
        """Returns MCP server information"""
        return {
            "name": "YouTube Summarizer MCP",
            "version": "1.0.0", 
            "description": "Access YouTube video summaries and AI-powered Q&A",
            "author": "Your Name",
            "license": "MIT",
            "capabilities": {
                "tools": list(self.tools.keys()),
                "resources": list(self.resources.keys())
            }
        }
    
    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            if tool_name == "search_videos":
                return self._search_videos(arguments)
            elif tool_name == "get_video_details":
                return self._get_video_details(arguments)
            elif tool_name == "ask_video_question":
                return self._ask_video_question(arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _search_videos(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search through videos using our existing semantic search"""
        from chatbot_qa import VideoQAChatbot
        
        chatbot = VideoQAChatbot()
        query = args.get("query", "")
        max_results = args.get("max_results", 5)
        
        # Use our existing search
        response = chatbot.ask_question(query, max_results)
        
        return {
            "results": response.get("sources", []),
            "total": len(response.get("sources", [])),
            "query": query
        }
    
    def _get_video_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get details for specific video"""
        from supabase_client import SupabaseDatabase
        
        db = SupabaseDatabase()
        video_id = args.get("video_id")
        
        # Find video by ID
        summaries = db.get_all_summaries()
        for summary in summaries:
            if summary.get("video_id") == video_id:
                return {
                    "video": summary,
                    "found": True
                }
        
        return {
            "error": f"Video {video_id} not found",
            "found": False
        }
    
    def _ask_video_question(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Ask question using our chatbot"""
        from chatbot_qa import VideoQAChatbot
        
        chatbot = VideoQAChatbot()
        question = args.get("question", "")
        
        response = chatbot.ask_question(question)
        
        return {
            "answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "question": question,
            "detected_topics": response.get("detected_topics", [])
        }

# MCP Server Configuration für Claude Code / andere Tools
MCP_CONFIG = {
    "mcpServers": {
        "YouTubeSummarizer": {
            "command": "python",
            "args": ["mcp_server.py"],
            "cwd": "D:/mcp"
        }
    }
}

def main():
    """Example MCP server usage"""
    server = YouTubeSummarizerMCP()
    
    print("YouTube Summarizer MCP Server")
    print("=" * 40)
    print(json.dumps(server.get_server_info(), indent=2))
    
    # Test tool call
    print("\nTesting search tool:")
    result = server.handle_tool_call("search_videos", {"query": "tower defense", "max_results": 3})
    print(f"Found {len(result.get('results', []))} videos")

if __name__ == "__main__":
    main()