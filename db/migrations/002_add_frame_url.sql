-- Add frame_url column to store link to best frame image in Supabase Storage
ALTER TABLE observations ADD COLUMN IF NOT EXISTS frame_url TEXT;
