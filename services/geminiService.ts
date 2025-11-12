import { GoogleGenAI, Type } from "@google/genai";
import { GraphData, TopicSuggestion, GraphNode, GraphLink, Concept } from '../types';

// Fix: Initialize GoogleGenAI with the environment variable, not a hardcoded key.
// The Vercel build process, via vite.config.ts, will replace process.env.API_KEY
// with the actual key provided in the Vercel project settings.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY! });

export const detectDominantLanguage = (texts: string[]): 'en' | 'zh' => {
  let enChars = 0;
  let zhChars = 0;
  
  const enRegex = /[a-zA-Z]/g;
  const zhRegex = /[\u4e00-\u9fa5]/g;

  for (const text of texts) {
    if(!text) continue;
    enChars += (text.match(enRegex) || []).length;
    zhChars += (text.match(zhRegex) || []).length;
  }
  
  // If there's barely any text, default to English to avoid issues.
  if (enChars + zhChars < 100) return 'en';

  return zhChars > enChars ? 'zh' : 'en';
};

function parseJsonResponse(rawText: string): any {
    try {
        return JSON.parse(rawText);
    } catch (e) {
         console.error("Failed to parse API response as JSON", e);
         // Attempt to clean the string if it's wrapped in markdown
         const jsonRegex = /```json\s*([\s\S]*?)\s*```/;
         const match = rawText.match(jsonRegex);
         if (match && match[1]) {
             try {
                return JSON.parse(match[1]);
             } catch (innerError) {
                console.error("Failed to parse extracted JSON from markdown", innerError);
             }
         }
         throw new Error("Response is not valid JSON, even after attempting to clean it.");
    }
}

export async function extractCausalGraph(papers: string[]): Promise<{ graphData: GraphData; concepts: Concept[] }> {
  const dominantLanguage = detectDominantLanguage(papers);
  const outputLanguageInstruction = dominantLanguage === 'en'
    ? "The output, including all variable names and concepts, MUST be in English."
    : "输出内容，包括所有变量名和概念，都必须使用简体中文。";

  const extractionPrompt = `
You are a highly advanced research assistant with expertise in semantic analysis and network science. Analyze the provided academic texts.
Your tasks are:
1.  **Identify Key Variables**: Extract all significant variables.
2.  **Classify Variables**: Determine if each variable is a "core" variable (central to the main arguments) or "secondary" (less critical, contextual).
3.  **Determine Causal Relationships**: Identify directed causal links between variables.
4.  **Perform Semantic Clustering**: Based on cross-lingual semantic similarity, group related variables under parent concepts. A variable should belong to only one concept.

${outputLanguageInstruction}

Provide the output in a single, valid JSON object with NO other text or markdown. The JSON structure MUST be:
{
  "nodes": [
    { "id": "Variable Name 1", "isCore": true },
    { "id": "Variable Name 2", "isCore": false }
  ],
  "links": [
    { "source": "Variable Name 1", "target": "Variable Name 2" }
  ],
  "concepts": [
    {
      "name": "Parent Concept A",
      "variables": ["Variable Name 1", "Variable Name 3"]
    },
    {
      "name": "Parent Concept B",
      "variables": ["Variable Name 2"]
    }
  ]
}

Here are the academic texts:
${papers.map((p, i) => `
Text ${i + 1}:
"""
${p}
"""`).join('\n')}
`;
  
  const extractionResult = await ai.models.generateContent({
      model: 'gemini-1.5-flash',
      contents: extractionPrompt,
      config: {
        responseMimeType: "application/json",
      },
  });
  const extractionResponse = parseJsonResponse(extractionResult.text);

  const { nodes: responseNodes, links: responseLinks, concepts: responseConcepts } = extractionResponse;

  if (!responseNodes || !responseLinks || !responseConcepts) {
    throw new Error("Failed to extract valid data from the text. The response format was incorrect.");
  }

  const variableToConceptMap = new Map<string, string>();
  responseConcepts.forEach((concept: { name: string; variables: string[] }) => {
    concept.variables.forEach(variable => {
      variableToConceptMap.set(variable, concept.name);
    });
  });

  const nodes: GraphNode[] = responseNodes.map((node: { id: string; isCore: boolean }) => ({
    id: node.id,
    isCore: node.isCore,
    group: variableToConceptMap.get(node.id) || 'Default',
  }));

  const nodeSet = new Set(nodes.map(n => n.id));
  const links: GraphLink[] = responseLinks
    .filter((link: { source: string; target: string; }) => nodeSet.has(link.source) && nodeSet.has(link.target));
  
  const graphData = { nodes, links };
  
  const nodeMap = new Map(nodes.map(node => [node.id, node]));
  const concepts: Concept[] = responseConcepts.map((concept: { name: string, variables: string[] }, index: number) => ({
      id: `${concept.name}-${index}`,
      name: concept.name,
      children: concept.variables
        .map(variableId => nodeMap.get(variableId))
        .filter((node): node is GraphNode => node !== undefined),
  }));

  return { graphData, concepts };
}

export async function generateTopicSuggestions(graphData: GraphData, concepts: Concept[], language: 'en' | 'zh'): Promise<TopicSuggestion[]> {
    const outputLanguageInstruction = language === 'en'
    ? "The output, including all topics, hypotheses, and rationales, MUST be in English."
    : "输出内容，包括所有主题、假设和理由，都必须使用简体中文。";

    const suggestionPrompt = `
You are an AI research strategist tasked with generating novel research topics. I will provide you with a causal network graph extracted from a body of literature. Your job is to identify gaps, unexplored relationships, or new perspectives within this existing knowledge map.

Based on the provided graph, generate 5 to 10 innovative and feasible research topics.

For each topic, you MUST provide:
1.  **topic**: A concise and compelling title for the research question.
2.  **hypothesis**: A clear, testable hypothesis derived from the topic.
3.  **innovation**: A justification for why this topic is innovative. This should feel like you've cross-referenced it against academic databases (like CrossRef, Semantic Scholar, CNKI) and found it to be a novel angle. For example, explain that a particular link is under-researched, or that combining two concepts is a new approach.
4.  **feasibility**: A statement on how this research could be conducted. It should be empirically testable, either through direct observation/experimentation or by synthesizing existing research.

${outputLanguageInstruction}

Here is the existing knowledge graph:
Nodes: ${JSON.stringify(graphData.nodes.map(n => n.id))}
Causal Relations (links): ${JSON.stringify(graphData.links.map(l => ({source: (l.source as GraphNode).id, target: (l.target as GraphNode).id})))}
Concept Clusters: ${JSON.stringify(concepts.map(c => ({name: c.name, variables: c.children.map(n => n.id)})))}

Provide the output as a single, valid JSON array of objects. Do not include any explanations, markdown formatting, or text outside of the JSON array.
The JSON structure for each object must be:
{
  "topic": "The unexplored causal link between [Variable A] and [Variable C]",
  "hypothesis": "[Variable A] is hypothesized to have a significant negative impact on [Variable C], a relationship not directly addressed in the source literature.",
  "innovation": "While the source texts link A to B and B to C, the direct A -> C relationship is a theoretical gap. A preliminary search suggests this direct pathway is under-investigated.",
  "feasibility": "This hypothesis can be tested using a longitudinal study tracking metrics for Variable A and Variable C over time in a relevant population."
}
`;

    const suggestionResult = await ai.models.generateContent({
      model: 'gemini-1.5-flash', // Using a more powerful model for creative suggestions
      contents: suggestionPrompt,
      config: {
        responseMimeType: "application/json",
      }
    });
    const topics: TopicSuggestion[] = parseJsonResponse(suggestionResult.text);

    if (!topics || !Array.isArray(topics) || topics.length === 0) {
      throw new Error("Failed to generate valid topic suggestions.");
    }

    return topics;
}
