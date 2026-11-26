import { LLMMessage, LLMProvider, LLMResponse, LLMToolDefinition } from '../types';
import { DEFAULT_MODELS } from '../constants';

export class GeminiProvider implements LLMProvider {
  constructor(private apiKey: string = '') {}

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['x-gemini-api-key'] = this.apiKey;
    }
    return headers;
  }

  private async postJson<T>(url: string, body: any): Promise<T> {
    const res = await fetch(url, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(body),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data?.error || `Request failed with status ${res.status}`);
    }
    return data as T;
  }

  async generateCompletion(
    messages: LLMMessage[],
    tools?: LLMToolDefinition[],
    systemInstruction?: string,
    modelName: string = DEFAULT_MODELS.text
  ): Promise<LLMResponse> {
    return this.postJson<LLMResponse>('/api/gemini/completion', {
      messages,
      tools,
      systemInstruction,
      modelName,
    });
  }

  async generateImage(
    prompt: string,
    modelName: string = DEFAULT_MODELS.image,
    onProgress?: (msg: string) => void,
    options: { aspectRatio?: string; imageSize?: string } = {},
    images?: string[]
  ): Promise<{ data: string; usage?: any }> {
    if (onProgress) onProgress('Generating image...');
    return this.postJson<{ data: string; usage?: any }>('/api/gemini/image', {
      prompt,
      modelName,
      options,
      images,
    });
  }

  async generateAudio(
    prompt: string,
    modelName: string = DEFAULT_MODELS.music,
    onProgress?: (msg: string) => void
  ): Promise<{ data: string; usage?: any }> {
    if (onProgress) onProgress('Generating audio...');
    return this.postJson<{ data: string; usage?: any }>('/api/gemini/audio', {
      prompt,
      modelName,
    });
  }

  async generateVideo(
    prompt: string,
    modelName: string = DEFAULT_MODELS.video,
    onProgress?: (msg: string) => void,
    options: {
      resolution?: '720p' | '1080p' | '4k';
      aspectRatio?: '16:9' | '9:16';
      durationSeconds?: 4 | 6 | 8;
    } = {},
    images?: string[]
  ): Promise<{ videoUrl: string; usage?: any }> {
    if (onProgress) onProgress('Generating video (this may take a minute)...');
    return this.postJson<{ videoUrl: string; usage?: any }>('/api/gemini/video', {
      prompt,
      modelName,
      options,
      images,
    });
  }
}
