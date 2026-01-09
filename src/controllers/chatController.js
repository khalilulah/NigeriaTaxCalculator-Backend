import { GoogleGenerativeAI } from "@google/generative-ai";
import mongoose from "mongoose";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
console.log("Gemini key loaded:", !!process.env.GEMINI_API_KEY);

// Use the same schema as setupDocuments.js
const DocumentChunkSchema = new mongoose.Schema({
  content: String,
  embedding: [Number],
  source: String,
  chunkIndex: Number,
  createdAt: { type: Date, default: Date.now },
});

const DocumentChunk = mongoose.model("DocumentChunk", DocumentChunkSchema);

// Function to generate embedding for user query
async function generateQueryEmbedding(query) {
  const model = genAI.getGenerativeModel({ model: "text-embedding-004" });
  const result = await model.embedContent(query);
  return result.embedding.values;
}

// Function to calculate cosine similarity
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// Function to find relevant document chunks
async function findRelevantChunks(queryEmbedding, topK = 5) {
  const results = await DocumentChunk.aggregate([
    {
      $vectorSearch: {
        index: "vector_index",
        path: "embedding",
        queryVector: queryEmbedding,
        numCandidates: 50,
        limit: topK,
      },
    },
    {
      $project: {
        content: 1,
        source: 1,
        score: { $meta: "vectorSearchScore" },
      },
    },
  ]);

  return results.map((chunk, i) => ({
    id: `SRC-${i + 1}`,
    content: chunk.content,
    source: chunk.source,
    similarity: chunk.score,
  }));
}

// Main chat function
export async function chat(req, res) {
  try {
    const { message } = req.body;

    if (!message || message.trim().length === 0) {
      return res.status(400).json({ error: "Message is required" });
    }

    console.log(`\nUser query: ${message}`);

    // Step 1: Generate embedding for user query
    const queryEmbedding = await generateQueryEmbedding(message);

    // Step 2: Find relevant document chunks
    const relevantChunks = await findRelevantChunks(queryEmbedding, 5);

    console.log(`Found ${relevantChunks.length} relevant chunks`);
    console.log(
      `Top similarity score: ${relevantChunks[0]?.similarity.toFixed(4)}`
    );

    // Step 3: Create a mapping of citation IDs to clean source names
    const sourceMap = {};
    relevantChunks.forEach((chunk, index) => {
      const citationId = `SRC-${index + 1}`;
      // Remove .pdf extension and clean up the filename
      const cleanSourceName = chunk.source.replace(/\.pdf$/i, "");
      sourceMap[citationId] = cleanSourceName;
    });

    // Step 3b: Build context from relevant chunks
    const context = relevantChunks
      .map(
        (chunk, index) => `
[SRC-${index + 1}]
Source: ${chunk.source.replace(/\.pdf$/i, "")}
Content:
${chunk.content}
`
      )
      .join("\n\n---\n\n");

    // Step 4: Create prompt for Gemini
    const prompt = `You are a domain-specific expert assistant that answers questions strictly using the provided documents about Nigeria’s tax laws and reforms.

Your task is to produce a deeply detailed, well-reasoned, and comprehensive explanation that fully leverages the provided sources. The response should read like an expert briefing or policy explanation rather than a summary.

The answer must naturally:
- establish the relevant background and context,
- explain the applicable laws, provisions, reforms, or rules in depth,
- and clearly articulate the practical implications, outcomes, or effects.

Do NOT label sections.
Do NOT mention or explain any framework, methodology, or reasoning process.

Context:
${context}

User question:
${message}

Core Rules (MANDATORY):
- Use ONLY the information provided in the context above.
- Do NOT introduce outside knowledge, assumptions, or interpretations.
- If the context does not contain enough information, respond exactly with:
  "I don't have enough information in the provided documents to answer that question."
- Every factual statement MUST end with a citation.
- If a statement cannot be clearly attributed to a source, do NOT include it.

Depth & Explanation Requirements:
- Provide a comprehensive and detailed answer where the context allows.
- Expand explanations when multiple related facts appear across different documents.
- Do NOT summarize if additional explanation improves clarity.
- Explain relationships, changes, and implications explicitly when supported by the documents.

Citation Rules (STRICT):
- Use ONLY the actual document names as citations.
- The source names available are: ${Object.values(sourceMap)
      .map((s) => "${s}")
      .join(", ")}
- NEVER use generic IDs such as [SRC-1], [Source 1], or similar.
- Citations must appear at the end of each factual sentence.

Formatting Rules:
- Use clear paragraphs separated by blank lines for each major idea.
- Use bullet points ONLY when listing multiple distinct items.
- Bullet points must start with a single asterisk and space: "* Item".
- Use **bold text** sparingly and ONLY for:
  - Act names
  - Statutory bodies
  - Key legal terms
- Use tables whenever comparing:
  - laws
  - tax rates
  - thresholds
  - dates
  - entities
- Do NOT include a table unless all information in it can be fully cited.
- Every table row must contain citations in the relevant cells or at the end of the row.

Style:
- Write in professional, precise, and authoritative language.
- use the STAR framework but Do NOT mention or explain any methodology, framework, or internal process



Answer:`;

    // Step 5: Generate response using Gemini (Direct REST API)
    const apiKey = process.env.GEMINI_API_KEY;
    const apiUrl = `https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key=${apiKey}`;

    const requestBody = {
      contents: [
        {
          parts: [
            {
              text: prompt,
            },
          ],
        },
      ],
    };

    const geminiResponse = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!geminiResponse.ok) {
      const errorText = await geminiResponse.text();
      throw new Error(
        `Gemini API error: ${geminiResponse.status} - ${errorText}`
      );
    }

    const geminiData = await geminiResponse.json();
    const response = geminiData.candidates[0].content.parts[0].text;

    console.log("Response generated successfully\n");

    // Return response with sources
    res.json({
      answer: response,
      sources: relevantChunks.map((chunk) => ({
        source: chunk.source.replace(/\.pdf$/i, ""), // Remove .pdf extension
        similarity: chunk.similarity.toFixed(4),
      })),
    });
  } catch (error) {
    console.error("Error in chat:", error);
    res
      .status(500)
      .json({ error: "An error occurred while processing your request" });
  }
}
