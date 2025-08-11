#!/usr/bin/env node
import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import rateLimit from 'express-rate-limit';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});

app.use(limiter);
app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

// Get Claude Desktop config path based on OS
function getClaudeConfigPath() {
  const platform = os.platform();
  const homeDir = os.homedir();
  
  switch (platform) {
    case 'darwin': // macOS
      return path.join(homeDir, 'Library', 'Application Support', 'Claude', 'claude_desktop_config.json');
    case 'win32': // Windows
      return path.join(os.homedir(), 'AppData', 'Roaming', 'Claude', 'claude_desktop_config.json');
    case 'linux': // Linux
      return path.join(homeDir, '.config', 'Claude', 'claude_desktop_config.json');
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
}

// API Routes
app.get('/api/config', async (req, res) => {
  try {
    const configPath = getClaudeConfigPath();
    let config = { mcpServers: {} };
    
    try {
      const configData = await fs.readFile(configPath, 'utf8');
      config = JSON.parse(configData);
    } catch (error) {
      // Config file doesn't exist or is invalid, return default
    }
    
    res.json({
      configPath,
      config,
      platform: os.platform()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/generate-config', async (req, res) => {
  try {
    const { apiKeys } = req.body;
    
    const config = {
      mcpServers: {}
    };

    // Perplexity Ask MCP Server
    if (apiKeys.perplexity) {
      config.mcpServers['perplexity-ask'] = {
        command: 'npx',
        args: ['-y', 'server-perplexity-ask'],
        env: {
          PERPLEXITY_API_KEY: apiKeys.perplexity
        }
      };
    }

    // DeepSeek MCP Server
    if (apiKeys.deepseek) {
      config.mcpServers['deepseek'] = {
        command: 'npx',
        args: ['-y', 'deepseek-mcp-server'],
        env: {
          DEEPSEEK_API_KEY: apiKeys.deepseek
        }
      };
    }

    // Gemini MCP Server
    if (apiKeys.gemini) {
      config.mcpServers['gemini'] = {
        type: 'stdio',
        command: 'npx',
        args: ['-y', 'github:aliargun/mcp-server-gemini'],
        env: {
          GEMINI_API_KEY: apiKeys.gemini
        }
      };
    }

    // Grok MCP Server
    if (apiKeys.grok) {
      config.mcpServers['grok-mcp'] = {
        command: 'npx',
        args: ['-y', 'grok-mcp'],
        env: {
          XAI_API_KEY: apiKeys.grok
        }
      };
    }

    // ElevenLabs (if we had an MCP server for it)
    if (apiKeys.elevenlabs) {
      config.mcpServers['elevenlabs'] = {
        command: 'npx',
        args: ['-y', 'elevenlabs-mcp-server'],
        env: {
          ELEVENLABS_API_KEY: apiKeys.elevenlabs
        }
      };
    }

    res.json({ config });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/save-config', async (req, res) => {
  try {
    const { config } = req.body;
    const configPath = getClaudeConfigPath();
    
    // Ensure directory exists
    const configDir = path.dirname(configPath);
    await fs.mkdir(configDir, { recursive: true });
    
    // Write config file
    await fs.writeFile(configPath, JSON.stringify(config, null, 2));
    
    res.json({ 
      success: true, 
      message: 'Configuration saved successfully',
      path: configPath 
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/server-status', (req, res) => {
  res.json({
    status: 'running',
    timestamp: new Date().toISOString(),
    platform: os.platform(),
    nodeVersion: process.version
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 MCP Dashboard running at http://localhost:${PORT}`);
  console.log(`📁 Claude config path: ${getClaudeConfigPath()}`);
  console.log(`💻 Platform: ${os.platform()}`);
});