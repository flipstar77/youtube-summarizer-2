# MCP Dashboard - API Key Manager

A beautiful, user-friendly dashboard for managing Model Context Protocol (MCP) server configurations and API keys.

## Features

🚀 **Complete MCP Integration**
- Perplexity AI (Real-time web search)
- DeepSeek AI (Advanced reasoning with R1/V3)
- Google Gemini (2.5 with thinking capabilities)
- Grok AI (xAI's latest models with vision)
- ElevenLabs (Text-to-speech)

🎯 **Smart Configuration**
- Automatic Claude Desktop config generation
- Cross-platform support (Windows, macOS, Linux)
- Secure API key handling
- One-click configuration export

💡 **User-Friendly Interface**
- Beautiful, responsive design
- Password fields with double-click reveal
- Real-time config preview
- Status notifications

## Quick Start

1. **Install Dependencies**
   ```bash
   cd mcp-dashboard
   npm install
   ```

2. **Start the Dashboard**
   ```bash
   npm start
   ```

3. **Open in Browser**
   ```
   http://localhost:3000
   ```

## How to Use

### 1. Enter API Keys
- **Perplexity**: Get from [Perplexity API](https://docs.perplexity.ai/guides/getting-started)
- **DeepSeek**: Get from [DeepSeek Platform](https://platform.deepseek.com/)
- **Gemini**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Grok**: Get from [xAI Console](https://console.x.ai/)
- **ElevenLabs**: Get from [ElevenLabs](https://elevenlabs.io/app/settings/api-keys)

### 2. Generate Configuration
Click "Generate Configuration" to create the Claude Desktop config automatically.

### 3. Save or Download
- **Save to Claude Desktop**: Automatically saves to the correct location
- **Download Config**: Download the JSON file manually

### 4. Restart Claude Desktop
After saving, restart Claude Desktop to load the new MCP servers.

## Configuration Locations

The dashboard automatically detects your platform and saves to:

- **Windows**: `%APPDATA%\\Claude\\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

## API Endpoints

- `GET /api/config` - Get current configuration
- `POST /api/generate-config` - Generate new configuration
- `POST /api/save-config` - Save configuration to Claude Desktop
- `GET /api/server-status` - Check server status

## Security Features

- Rate limiting (100 requests per 15 minutes)
- Password input fields
- Local-only operation (no external data transmission)
- Secure file handling

## Supported MCP Servers

| Server | Tool | Capabilities |
|--------|------|-------------|
| **Perplexity Ask** | `perplexity_ask`, `perplexity_research`, `perplexity_reason` | Web search, research, reasoning |
| **DeepSeek** | `chat_completion`, `multi_turn_chat` | R1 reasoning, V3 chat, fallback |
| **Gemini** | `generate_text`, `analyze_image`, `embeddings` | 2.5 thinking, vision, JSON mode |
| **Grok** | `chat_completion`, `image_understanding`, `function_calling` | Grok 3, vision, reasoning |
| **ElevenLabs** | TTS tools | Voice synthesis, audio generation |

## Development

### Project Structure
```
mcp-dashboard/
├── server.js          # Express server
├── package.json       # Dependencies
├── public/
│   └── index.html     # Dashboard interface
└── README.md          # This file
```

### Running in Development
```bash
npm run dev
```

### Environment Variables
- `PORT`: Server port (default: 3000)

## License

MIT License - feel free to use and modify as needed.

## Troubleshooting

1. **Dashboard won't start**: Check Node.js version (requires 16+)
2. **Config not saving**: Ensure Claude Desktop directory exists
3. **API keys not working**: Verify keys are valid and have proper permissions
4. **MCP servers not appearing**: Restart Claude Desktop after saving config

## Support

For issues with specific MCP servers, refer to their respective repositories:
- [Perplexity Ask MCP](https://github.com/ppl-ai/modelcontextprotocol)
- [DeepSeek MCP](https://github.com/DMontgomery40/deepseek-mcp-server)
- [Gemini MCP](https://github.com/aliargun/mcp-server-gemini)
- [Grok MCP](https://github.com/Bob-lance/grok-mcp)