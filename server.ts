import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express, { Request, Response } from 'express';
import { FunctionDeclaration, GoogleGenAI, Tool, Type } from '@google/genai';
import { DEFAULT_MODELS } from './src/core/llm/constants';
import { calculateTokensForCost } from './src/core/llm/pricing';
import { LLMMessage, LLMToolCall, LLMToolDefinition } from './src/core/llm/types';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const videoDownloadMap = new Map<string, { uri: string; apiKey: string }>();

function getApiKey(req: Request): string | undefined {
  const headerKey = req.headers['x-gemini-api-key'];
  if (typeof headerKey === 'string' && headerKey.trim()) {
    return headerKey.trim();
  }
  return process.env.GEMINI_API_KEY?.trim() || undefined;
}

function mapMessagesToGemini(messages: LLMMessage[]): any[] {
  return messages
    .filter((m) => m.role !== 'system')
    .map((m) => {
      const role = m.role === 'assistant' ? 'model' : 'user';
      const parts: any[] = [];

      if (m.content) {
        parts.push({ text: m.content });
      }

      if (m.tool_calls) {
        for (const tc of m.tool_calls) {
          parts.push({
            functionCall: {
              name: tc.function.name,
              args: JSON.parse(tc.function.arguments),
            },
          });
        }
      }

      if (m.role === 'tool' && m.name) {
        parts.push({
          functionResponse: {
            name: m.name,
            response: JSON.parse(m.content),
          },
        });
      }

      if (m.images) {
        for (const img of m.images) {
          const base64Match = img.match(/^data:(image\/[a-z]+);base64,(.+)$/);
          if (base64Match) {
            parts.push({
              inlineData: {
                mimeType: base64Match[1],
                data: base64Match[2],
              },
            });
          } else {
            parts.push({
              inlineData: {
                mimeType: 'image/jpeg',
                data: img,
              },
            });
          }
        }
      }

      return { role, parts };
    });
}

function mapToGeminiSchema(schema: any): any {
  if (!schema) return undefined;

  const typeStr = (schema.type || 'string').toUpperCase();
  const mappedType = Type[typeStr as keyof typeof Type] || Type.STRING;

  const result: any = {
    type: mappedType,
    description: schema.description,
    nullable: schema.nullable,
    minItems: schema.minItems,
    maxItems: schema.maxItems,
    minimum: schema.minimum,
    maximum: schema.maximum,
    minLength: schema.minLength,
    maxLength: schema.maxLength,
  };

  if (schema.properties) {
    result.properties = Object.keys(schema.properties).reduce((acc, key) => {
      acc[key] = mapToGeminiSchema(schema.properties[key]);
      return acc;
    }, {} as Record<string, any>);
  }

  if (schema.required) {
    result.required = schema.required;
  }

  if (schema.items) {
    result.items = mapToGeminiSchema(schema.items);
  }

  if (schema.enum) {
    result.enum = schema.enum;
  }

  return result;
}

async function pollVideoOperation(
  client: GoogleGenAI,
  apiKey: string,
  operation: any,
  modelName: string
) {
  while (!operation.done) {
    await new Promise((resolve) => setTimeout(resolve, 10000));
    operation = await (client as any).operations.getVideosOperation({
      operation,
    });
  }

  const videoData = operation.response?.generatedVideos?.[0];
  let videoUri = (videoData?.video as any)?.uri || '';

  if (videoUri && videoUri.includes('generativelanguage.googleapis.com')) {
    const id = crypto.randomUUID();
    videoDownloadMap.set(id, { uri: videoUri, apiKey });
    videoUri = `/api/gemini/video-download/${id}`;
  }

  const videoDuration = videoData?.durationSeconds || 4;
  const videoTokens = calculateTokensForCost(modelName, videoDuration);

  return {
    videoUrl: videoUri,
    usage: {
      promptTokens: 0,
      completionTokens: videoTokens,
      totalTokens: videoTokens,
      duration: videoDuration,
    },
  };
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '50mb' }));

  app.post('/api/gemini/completion', async (req: Request, res: Response) => {
    try {
      const apiKey = getApiKey(req);
      if (!apiKey) {
        res.status(401).json({ error: 'Gemini API key is required' });
        return;
      }

      const {
        messages,
        tools,
        systemInstruction,
        modelName = DEFAULT_MODELS.text,
      } = req.body as {
        messages: LLMMessage[];
        tools?: LLMToolDefinition[];
        systemInstruction?: string;
        modelName?: string;
      };

      const client = new GoogleGenAI({ apiKey });
      const contents = mapMessagesToGemini(messages || []);
      const systemTools: Tool[] | undefined = tools
        ? [
            {
              functionDeclarations: tools.map(
                (t) =>
                  ({
                    name: t.function.name,
                    description: t.function.description,
                    parameters: mapToGeminiSchema(t.function.parameters),
                  } as FunctionDeclaration)
              ),
            },
          ]
        : undefined;

      const result = await client.models.generateContent({
        model: modelName,
        contents,
        config: {
          systemInstruction,
          tools: systemTools,
        },
      });

      const candidate = result.candidates?.[0];
      const parts = candidate?.content?.parts || [];

      let contentStr: string | null = null;
      const toolCalls: LLMToolCall[] = [];

      for (const part of parts) {
        if (part.text) {
          contentStr = (contentStr || '') + part.text;
        }
      }

      if (candidate?.content?.parts) {
        for (const part of candidate.content.parts) {
          if (part.functionCall) {
            toolCalls.push({
              id: Math.random().toString(36).substring(7),
              type: 'function',
              function: {
                name: part.functionCall.name || '',
                arguments: JSON.stringify(part.functionCall.args),
              },
            });
          }
        }
      }

      if (result.functionCalls && toolCalls.length === 0) {
        for (const call of result.functionCalls) {
          toolCalls.push({
            id: Math.random().toString(36).substring(7),
            type: 'function',
            function: {
              name: call.name || '',
              arguments: JSON.stringify(call.args),
            },
          });
        }
      }

      const usage = result.usageMetadata
        ? {
            promptTokens: result.usageMetadata.promptTokenCount || 0,
            completionTokens:
              (result.usageMetadata.candidatesTokenCount || 0) +
              (result.usageMetadata.thoughtsTokenCount || 0),
            totalTokens: result.usageMetadata.totalTokenCount || 0,
          }
        : undefined;

      res.json({
        content: contentStr,
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        usage,
        finishReason: candidate?.finishReason as string,
        raw: result,
        request: {
          contents,
          systemInstruction,
          tools: systemTools,
        },
      });
    } catch (error: any) {
      console.error('[/api/gemini/completion] Error:', error);
      res.status(error?.status || 500).json({
        error: error?.message || 'Failed to generate completion',
      });
    }
  });

  app.post('/api/gemini/image', async (req: Request, res: Response) => {
    try {
      const apiKey = getApiKey(req);
      if (!apiKey) {
        res.status(401).json({ error: 'Gemini API key is required' });
        return;
      }

      const {
        prompt,
        modelName = DEFAULT_MODELS.image,
        options = {},
        images,
      } = req.body as {
        prompt: string;
        modelName?: string;
        options?: { aspectRatio?: string; imageSize?: string };
        images?: string[];
      };

      const client = new GoogleGenAI({ apiKey });
      const config = {
        responseModalities: ['IMAGE', 'TEXT'],
        imageConfig: {
          aspectRatio: options.aspectRatio || '16:9',
          imageSize: options.imageSize || '1K',
        },
      };

      const contents: any[] = [{ text: prompt }];

      if (images && images.length > 0) {
        for (const img of images) {
          const base64Match = img.match(/^data:(image\/[a-z]+);base64,(.+)$/);
          if (base64Match) {
            contents.push({
              inlineData: {
                mimeType: base64Match[1],
                data: base64Match[2],
              },
            });
          }
        }
      }

      const result = await client.models.generateContent({
        model: modelName,
        contents,
        config: config as any,
      });

      const candidate = result.candidates?.[0];
      const parts = candidate?.content?.parts || [];
      let base64Data: string | undefined;

      for (const part of parts) {
        if (part.inlineData) {
          base64Data = part.inlineData.data;
        }
      }

      const imageTokens = calculateTokensForCost(modelName, 1);

      res.json({
        data: base64Data || '',
        usage: {
          promptTokens: result.usageMetadata?.promptTokenCount || 0,
          completionTokens:
            (result.usageMetadata?.candidatesTokenCount || 0) + imageTokens,
          totalTokens:
            (result.usageMetadata?.totalTokenCount || 0) + imageTokens,
          count: 1,
        },
      });
    } catch (error: any) {
      console.error('[/api/gemini/image] Error:', error);
      res.status(error?.status || 500).json({
        error: error?.message || 'Failed to generate image',
      });
    }
  });

  app.post('/api/gemini/audio', async (req: Request, res: Response) => {
    try {
      const apiKey = getApiKey(req);
      if (!apiKey) {
        res.status(401).json({ error: 'Gemini API key is required' });
        return;
      }

      const { prompt, modelName = DEFAULT_MODELS.music } = req.body as {
        prompt: string;
        modelName?: string;
      };

      const client = new GoogleGenAI({ apiKey });
      const result = await client.models.generateContent({
        model: modelName,
        contents: prompt,
        config: {
          responseModalities: ['AUDIO', 'TEXT'],
        },
      });

      const candidate = result.candidates?.[0];
      const parts = candidate?.content?.parts || [];
      let base64Data: string | undefined;

      for (const part of parts) {
        if (part.inlineData) {
          base64Data = part.inlineData.data;
        }
      }

      const audioTokens = calculateTokensForCost(modelName, 1);

      res.json({
        data: base64Data || '',
        usage: {
          promptTokens: result.usageMetadata?.promptTokenCount || 0,
          completionTokens:
            (result.usageMetadata?.candidatesTokenCount || 0) + audioTokens,
          totalTokens:
            (result.usageMetadata?.totalTokenCount || 0) + audioTokens,
          count: 1,
        },
      });
    } catch (error: any) {
      console.error('[/api/gemini/audio] Error:', error);
      res.status(error?.status || 500).json({
        error: error?.message || 'Failed to generate audio',
      });
    }
  });

  app.post('/api/gemini/video', async (req: Request, res: Response) => {
    try {
      const apiKey = getApiKey(req);
      if (!apiKey) {
        res.status(401).json({ error: 'Gemini API key is required' });
        return;
      }

      const {
        prompt,
        modelName = DEFAULT_MODELS.video,
        options = {},
        images,
      } = req.body as {
        prompt: string;
        modelName?: string;
        options?: {
          resolution?: '720p' | '1080p' | '4k';
          aspectRatio?: '16:9' | '9:16';
          durationSeconds?: 4 | 6 | 8;
        };
        images?: string[];
      };

      const client = new GoogleGenAI({ apiKey });
      const videoConfig: any = {
        resolution: options.resolution || '720p',
        aspectRatio: options.aspectRatio || '16:9',
        durationSeconds: options.durationSeconds || 4,
        sampleCount: 1,
      };

      if (modelName.includes('lite')) {
        const request: any = {
          model: modelName,
          prompt,
          config: videoConfig,
        };

        if (images && images.length > 0) {
          const m = images[0].match(/^data:(image\/[a-z]+);base64,(.+)$/);
          if (m) {
            request.image = {
              imageBytes: m[2],
              mimeType: m[1],
            };
          }
        }

        const operation = await (client.models as any).generateVideos(request);
        const result = await pollVideoOperation(client, apiKey, operation, modelName);
        res.json(result);
        return;
      }

      const generateVideoPayload: any = {
        model: modelName,
        config: videoConfig,
      };

      if (prompt) {
        generateVideoPayload.prompt = prompt;
      }

      if (images && images.length > 0) {
        const referenceImagesPayload: any[] = [];
        for (const img of images) {
          const m = img.match(/^data:(image\/[a-z]+);base64,(.+)$/);
          if (m) {
            referenceImagesPayload.push({
              image: {
                imageBytes: m[2],
                mimeType: m[1],
              },
              referenceType: 'asset',
            });
          }
        }

        if (referenceImagesPayload.length > 0) {
          generateVideoPayload.config.referenceImages = referenceImagesPayload;
          videoConfig.durationSeconds = 8;
        }
      }

      if (videoConfig.resolution === '1080p' || videoConfig.resolution === '4k') {
        videoConfig.durationSeconds = 8;
      }

      const operation = await (client.models as any).generateVideos(generateVideoPayload);
      const result = await pollVideoOperation(client, apiKey, operation, modelName);
      res.json(result);
    } catch (error: any) {
      console.error('[/api/gemini/video] Error:', error);
      res.status(error?.status || 500).json({
        error: error?.message || 'Failed to generate video',
      });
    }
  });

  app.get('/api/gemini/video-download/:id', async (req: Request, res: Response) => {
    const id = Array.isArray(req.params.id) ? req.params.id[0] : req.params.id;
    const entry = videoDownloadMap.get(id);
    if (!entry) {
      res.status(404).json({ error: 'Video not found' });
      return;
    }

    try {
      const separator = entry.uri.includes('?') ? '&' : '?';
      const upstream = await fetch(`${entry.uri}${separator}key=${entry.apiKey}`);
      if (!upstream.ok) {
        res.status(upstream.status).send('Failed to fetch video');
        return;
      }
      const contentType = upstream.headers.get('content-type') || 'video/mp4';
      res.setHeader('Content-Type', contentType);
      const buffer = Buffer.from(await upstream.arrayBuffer());
      res.send(buffer);
    } catch (error) {
      console.error('[/api/gemini/video-download] Error:', error);
      res.status(500).send('Failed to download video');
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.use((_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${port}`);
  });
}

startServer();
