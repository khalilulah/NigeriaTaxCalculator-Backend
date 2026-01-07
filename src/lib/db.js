// src/lib/db.ts
import mongoose from "mongoose";

const connectDb = async () => {
  try {
    const mongoUri = process.env.MONGO_URI;

    // DEBUG: Check if env variable is loaded
    console.log("🔍 MONGO_URI exists:", !!mongoUri);
    console.log(
      "🔍 MONGO_URI value:",
      mongoUri ? mongoUri.substring(0, 20) + "..." : "UNDEFINED"
    );

    if (!mongoUri) {
      throw new Error("MONGO_URI is not defined in .env file");
    }

    console.log("🔄 Connecting to MongoDB...");
    await mongoose.connect(mongoUri);
    console.log("✅ MongoDB connected successfully");
  } catch (error) {
    console.error("❌ MongoDB connection error:", error);
    process.exit(1);
  }
};

export default connectDb;
