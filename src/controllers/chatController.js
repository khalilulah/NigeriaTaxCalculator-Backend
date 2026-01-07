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
  // Get all chunks from database
  const allChunks = await DocumentChunk.find({});

  // Calculate similarity scores
  const chunksWithScores = allChunks.map((chunk) => ({
    content: chunk.content,
    source: chunk.source,
    similarity: cosineSimilarity(queryEmbedding, chunk.embedding),
  }));

  // Sort by similarity and return top K
  return chunksWithScores
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
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

    // Step 3: Build context from relevant chunks
    const context = relevantChunks
      .map((chunk, i) => `[Source ${i + 1}: ${chunk.source}]\n${chunk.content}`)
      .join("\n\n---\n\n");

    // Step 4: Create prompt for Gemini
    const prompt = `You are a helpful assistant that answers questions based on the provided documents about Nigeria's tax laws and reforms.

Context from documents:
${context}

User question: ${message}

Instructions:
- Answer the question using the STAR framework where applicable:
  * Situation: Provide background/context from the documents
  * Task: Explain what needs to be understood or addressed
  * Action: Describe the specific provisions, changes, or steps outlined in the tax laws
  * Result: Explain the expected outcomes, impacts, or implications
- Answer based ONLY on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer that question."
- Be specific and cite which source the information comes from when relevant
- Keep your answer clear, well-structured, and concise
- Use the STAR format naturally - don't force it if the question is simple and doesn't need this structure

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
        source: chunk.source,
        similarity: chunk.similarity.toFixed(4),
      })),
    });

    console.log("Response generated successfully\n");

    // Return response with sources
  } catch (error) {
    console.error("Error in chat:", error);
    res
      .status(500)
      .json({ error: "An error occurred while processing your request" });
  }
}
