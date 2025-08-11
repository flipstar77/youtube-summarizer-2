# Vector Database Setup Instructions

Your vector database setup is almost complete! Follow these steps to finish:

## Step 1: Apply SQL Schema to Supabase

1. **Go to your Supabase Dashboard:**
   - Visit: https://app.supabase.com/
   - Open your project
   - Go to the SQL Editor tab

2. **Copy and Paste the SQL Commands:**
   - Open the file `manual_vector_setup.sql` in your project folder
   - Copy all the SQL commands (the entire file)
   - Paste them into the Supabase SQL Editor
   - Click "Run" to execute all commands

3. **Verify the Schema:**
   - You should see messages like "CREATE EXTENSION", "ALTER TABLE", etc.
   - If you see any errors, they're likely just notices (extension already exists, etc.)

## Step 2: Generate Embeddings for Your Existing Summaries

After applying the SQL schema, run this command:

```bash
python setup_vector_database.py
```

This script will:
- ✅ Test if the schema was applied correctly  
- 🔄 Generate OpenAI embeddings for all 18+ existing summaries
- 🔍 Test vector search functionality
- 📊 Show you database statistics

## What This Gives You

Once complete, your YouTube summarizer will have:

- **Semantic Search**: Find similar videos based on content meaning, not just keywords
- **Better Categorization**: Group similar topics together automatically  
- **Content Recommendations**: Show related summaries on each video page
- **Fix for "0 data" Issue**: Your database will show proper content counts

## Expected Results

You should see output like:
```
✅ Vector schema detected - found 18 summaries needing embeddings
📝 Processing 18 summaries...
  [1/18] Processing: How AI Works in 2024...
    ✅ Added 1536-dimensional embedding
...
🎉 Completed! 18/18 summaries now have embeddings
✅ Vector search working - found 3 similar summaries
✅ SUCCESS: Vector database is fully set up!
```

## Troubleshooting

**If you see schema errors:**
- Make sure you copied the entire `manual_vector_setup.sql` file
- Run the SQL commands in Supabase one by one if needed

**If embeddings fail:**
- Check that OPENAI_API_KEY is set in your `.env` file
- Make sure you have OpenAI API credits available

**Need help?** The script will give you specific error messages and solutions.