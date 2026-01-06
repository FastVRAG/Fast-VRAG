import argparse
import logging

from chromadb_indexer import ChromaDBIndexer
from config.build_index_config import *

def main():
    parser = argparse.ArgumentParser(description='Build ChromaDB index from Wikipedia JSON')
    parser.add_argument(
        '--wiki-json',
        type=str,
        default=KNOWLEDGE_JSON_PATH,
        help='Path to Wikipedia JSON file'
    )
    parser.add_argument(
        '--chroma-db-path',
        type=str,
        default=CHROMA_DB_PATH,
        help='Path to ChromaDB storage'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Recreate collections (delete existing)'
    )
    parser.add_argument(
        '--clip-collection',
        type=str,
        default=CLIP_COLLECTION_NAME,
        help='Name for CLIP collection'
    )

    args = parser.parse_args()

    # Initialize indexer
    indexer = ChromaDBIndexer(
        chroma_db_path=args.chroma_db_path,
        clip_collection_name=args.clip_collection,
        recreate_collections=args.recreate
    )

    # Build index
    indexer.build_index_from_json(
        wiki_json_path=args.wiki_json,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()