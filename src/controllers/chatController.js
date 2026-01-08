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
    const prompt = `You are a domain-specific assistant that answers questions strictly using the provided documents about Nigeria's tax laws and reforms.

Use the information in the context to produce a clear, logical, and well-reasoned answer. Structure your response so that it naturally:
- establishes the relevant background,
- explains the applicable rule, action, or provision,
- and concludes with the outcome or implication,

but do NOT label sections or mention any reasoning framework.

Context:
${context}

User question:
${message}

Rules:
- Use ONLY the information provided in the context above
- Do NOT introduce outside knowledge, assumptions, or interpretations
- If the context does not contain enough information, respond exactly with:
  "I don't have enough information in the provided documents to answer that question."
- Every factual statement MUST end with a citation in square brackets using the SOURCE NAME (not the ID)
- The source names available are: ${Object.values(sourceMap)
      .map((s) => `"${s}"`)
      .join(", ")}
- Format citations like this: [Nigeria Tax Act 2025] or [Joint Revenue Board Act]
- NEVER use citation IDs like [SRC-1] or [Source 1] - always use the actual document name
- If a fact cannot be attributed to a source, do not include it
- Write in professional, clear, and concise language
- Do NOT mention or explain any framework or methodology used

Answer:`;

    // Step 5: Generate response using Gemini (Direct REST API)
    const apiKey = process.env.GEMINI_API_KEY;
    const apiUrl = `https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent?key=${apiKey}`;

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
