-- DSPy Supabase RAG Schema
-- Run this in your Supabase SQL Editor (https://supabase.com/dashboard/project/YOUR_PROJECT/sql)

-- Enable the vector extension
create extension if not exists vector with schema extensions;

-- Create the documents table
-- Using 1536 dimensions (Supabase HNSW limit is 2000)
-- Works with: text-embedding-3-small, or text-embedding-3-large with dimensions=1536
create table documents (
  id bigint primary key generated always as identity,
  content text not null,
  source text,
  section text,
  metadata jsonb default '{}'::jsonb,
  embedding extensions.vector(1536)
);

-- Create an HNSW index for fast similarity search
create index on documents using hnsw (embedding vector_cosine_ops);

-- Enable Row Level Security (optional but recommended)
alter table documents enable row level security;

-- Create a policy to allow all operations (adjust for production)
create policy "Allow all" on documents for all using (true);

-- Create a function for similarity search
create or replace function match_documents (
  query_embedding extensions.vector(1536),
  match_threshold float default 0.5,
  match_count int default 5
)
returns table (
  id bigint,
  content text,
  source text,
  section text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    documents.id,
    documents.content,
    documents.source,
    documents.section,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;
