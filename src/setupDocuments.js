import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { PDFExtract } from "pdf.js-extract";
import { GoogleGenerativeAI } from "@google/generative-ai";
import mongoose from "mongoose";
import { config } from "dotenv";

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env from root directory
config({ path: path.join(__dirname, "..", ".env") });

// Debug: Check if env variables are loaded
console.log("MongoDB URI loaded:", process.env.MONGO_URI ? "Yes ✓" : "No ✗");
console.log(
  "Gemini API Key loaded:",
  process.env.GEMINI_API_KEY ? "Yes ✓" : "No ✗"
);
console.log("");

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const pdfExtract = new PDFExtract();

// Define a schema for storing document chunks
const DocumentChunkSchema = new mongoose.Schema({
  content: String,
  embedding: [Number],
  source: String,
  chunkIndex: Number,
  createdAt: { type: Date, default: Date.now },
});

const DocumentChunk = mongoose.model("DocumentChunk", DocumentChunkSchema);

// Function to extract text from PDF
async function extractTextFromPDF(filePath) {
  try {
    const data = await pdfExtract.extract(filePath, {});

    // Extract text from all pages
    let fullText = "";
    data.pages.forEach((page) => {
      page.content.forEach((item) => {
        if (item.str) {
          fullText += item.str + " ";
        }
      });
      fullText += "\n"; // Add newline between pages
    });

    return fullText.trim();
  } catch (error) {
    console.error("Error extracting text from PDF:", error);
    return "";
  }
}

// Function to split text into chunks
function splitIntoChunks(text, chunkSize = 500) {
  const words = text.split(/\s+/).filter((word) => word.trim().length > 0);
  const chunks = [];

  for (let i = 0; i < words.length; i += chunkSize) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    if (chunk.trim().length > 0) {
      chunks.push(chunk);
    }
  }

  return chunks;
}

// Function to generate embeddings using Gemini
async function generateEmbedding(text) {
  try {
    const model = genAI.getGenerativeModel({ model: "text-embedding-004" });
    const result = await model.embedContent(text);
    return result.embedding.values;
  } catch (error) {
    console.error("Error generating embedding:", error);
    throw error;
  }
}

// Function to process a single PDF
async function processPDF(filePath) {
  console.log(`\nProcessing: ${path.basename(filePath)}`);

  try {
    // Extract text from PDF
    const text = await extractTextFromPDF(filePath);
    console.log(`Extracted ${text.length} characters`);

    if (text.length === 0) {
      console.log(
        `⚠️  Warning: No text extracted from ${path.basename(filePath)}`
      );
      console.log(`   This PDF might be scanned images or encrypted.`);
      return;
    }

    // Split into chunks
    const chunks = splitIntoChunks(text);
    console.log(`Created ${chunks.length} chunks`);

    if (chunks.length === 0) {
      console.log(
        `⚠️  Warning: No chunks created from ${path.basename(filePath)}`
      );
      return;
    }

    // Process each chunk
    const fileName = path.basename(filePath);

    for (let i = 0; i < chunks.length; i++) {
      console.log(`Processing chunk ${i + 1}/${chunks.length}...`);

      // Generate embedding
      const embedding = await generateEmbedding(chunks[i]);

      // Save to MongoDB
      await DocumentChunk.create({
        content: chunks[i],
        embedding: embedding,
        source: fileName,
        chunkIndex: i,
      });

      // Small delay to avoid rate limits
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    console.log(`✓ Completed: ${fileName}`);
  } catch (error) {
    console.error(
      `Error processing ${path.basename(filePath)}:`,
      error.message
    );
  }
}

// Main function
async function main() {
  try {
    // Connect to MongoDB
    await mongoose.connect(process.env.MONGO_URI);
    console.log("Connected to MongoDB\n");

    // Clear existing documents (optional - remove if you want to keep old data)
    await DocumentChunk.deleteMany({});
    console.log("Cleared existing documents\n");

    // Path to your PDFs folder
    const pdfFolder = path.join(__dirname, "..", "pdfs");

    // Create folder if it doesn't exist
    if (!fs.existsSync(pdfFolder)) {
      fs.mkdirSync(pdfFolder);
      console.log('Created "pdfs" folder. Please add your PDF files there.');
      process.exit(0);
    }

    // Get all PDF files
    const files = fs
      .readdirSync(pdfFolder)
      .filter((file) => file.toLowerCase().endsWith(".pdf"))
      .map((file) => path.join(pdfFolder, file));

    if (files.length === 0) {
      console.log('No PDF files found in the "pdfs" folder.');
      process.exit(0);
    }

    console.log(`Found ${files.length} PDF file(s)`);

    // Process each PDF
    for (const file of files) {
      await processPDF(file);
    }

    const totalChunks = await DocumentChunk.countDocuments();
    console.log(`\n${"=".repeat(50)}`);
    console.log(`✓ All documents processed successfully!`);
    console.log(`📊 Total chunks stored in database: ${totalChunks}`);
    console.log(`${"=".repeat(50)}`);
  } catch (error) {
    console.error("Error:", error);
  } finally {
    await mongoose.connection.close();
    console.log("\nDatabase connection closed");
  }
}

// Run the script
main();
