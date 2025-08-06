"""Micro-ingestion functionality for incremental updates"""

import logging
import time
from collections import Counter

logger = logging.getLogger('FuzzyShell')


def micro_ingest(fuzzyshell):
    """
    Perform a micro-ingest of new commands since last update.
    
    This is designed to be fast and run on startup to catch new commands
    without needing a full re-ingestion.
    
    Returns:
        tuple: (new_commands_count, should_rebuild_ann)
    """
    start_time = time.time()
    
    # Get last update timestamp
    last_updated = fuzzyshell.get_metadata('last_micro_ingest', '1970-01-01 00:00:00')
    
    # Get count when ANN was last built
    ann_command_count = int(fuzzyshell.get_metadata('ann_command_count', '0'))
    current_command_count = fuzzyshell.get_indexed_count()
    
    # Read recent history entries
    history_file = fuzzyshell.get_shell_history_file()
    new_commands = []
    
    try:
        with open(history_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Read last 500 lines for efficiency
            lines = f.readlines()[-500:]
            
        for line in lines:
            raw_command = line.strip()
            if not raw_command:
                continue
                
            command = fuzzyshell.clean_shell_command(raw_command)
            if not command:
                continue
                
            # Check if this command already exists using DAL
            existing_ids = fuzzyshell.command_dal.get_command_ids_for_commands([command])
            if command not in existing_ids:
                new_commands.append(command)
                
    except FileNotFoundError:
        logger.warning("History file not found for micro-ingest")
        return 0, False
    
    # Process new commands
    if new_commands:
        logger.info("Micro-ingest found %d new commands", len(new_commands))
        
        # Initialize model if needed
        if fuzzyshell._model is None:
            fuzzyshell.init_model_sync()
        
        poorly_clustered_count = 0
        
        for command in new_commands:
            # Process terms for BM25
            terms = fuzzyshell.tokenize(command)
            term_freq = Counter(terms)
            
            # Use DAL to add command with terms
            command_id = fuzzyshell.command_dal.add_single_command_with_terms(command, terms)
            
            # Generate embedding
            embedding = fuzzyshell.model.encode([command])[0]
            if len(embedding) > 384:  # MODEL_OUTPUT_DIM
                embedding = embedding[:384]
            
            # Track if this embedding is far from existing clusters
            if fuzzyshell.ann_index and fuzzyshell.ann_index.is_trained:
                # Check distance to nearest cluster
                distances = fuzzyshell.ann_index.get_cluster_distances(embedding)
                min_distance = min(distances) if distances else float('inf')
                
                # If very far from any cluster, it's poorly clustered
                if min_distance > 0.8:  # Threshold for "poor fit"
                    poorly_clustered_count += 1
            
            # Store embedding using DAL
            embedding_quantized = fuzzyshell.quantize_embedding(embedding)
            fuzzyshell.command_dal.add_embeddings_batch([(command_id, embedding_quantized.tobytes())])
        
        # DAL handles commits automatically
        fuzzyshell.update_corpus_stats()
        
        # Update metadata
        fuzzyshell.set_metadata('last_micro_ingest', time.strftime('%Y-%m-%d %H:%M:%S'))
        fuzzyshell._update_item_count()
        
        # Track poorly clustered commands
        existing_poorly_clustered = int(fuzzyshell.get_metadata('poorly_clustered_commands', '0'))
        fuzzyshell.set_metadata('poorly_clustered_commands', 
                               str(existing_poorly_clustered + poorly_clustered_count))
        
        elapsed = time.time() - start_time
        logger.info("Micro-ingest completed in %.2fs", elapsed)
    else:
        logger.debug("No new commands found in micro-ingest")
        fuzzyshell.set_metadata('last_micro_ingest', time.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Determine if ANN rebuild is needed
    new_command_count = fuzzyshell.get_indexed_count()
    commands_since_ann = new_command_count - ann_command_count
    poorly_clustered = int(fuzzyshell.get_metadata('poorly_clustered_commands', '0'))
    
    # Rebuild if:
    # 1. More than 10% new commands since last ANN build
    # 2. More than 100 new commands (absolute)
    # 3. More than 5% of new commands are poorly clustered
    should_rebuild = False
    rebuild_reason = None
    
    if ann_command_count > 0:
        growth_ratio = commands_since_ann / ann_command_count
        if growth_ratio > 0.1:
            should_rebuild = True
            rebuild_reason = f"{growth_ratio*100:.1f}% growth since last ANN build"
    
    if commands_since_ann > 100:
        should_rebuild = True
        rebuild_reason = f"{commands_since_ann} new commands since last ANN build"
    
    if poorly_clustered > commands_since_ann * 0.05 and poorly_clustered > 10:
        should_rebuild = True
        rebuild_reason = f"{poorly_clustered} poorly clustered commands"
    
    if should_rebuild and rebuild_reason:
        logger.info("ANN rebuild recommended: %s", rebuild_reason)
    
    return len(new_commands), should_rebuild


def suggest_ann_rebuild(fuzzyshell):
    """Check if ANN rebuild is recommended and return suggestion message."""
    ann_command_count = int(fuzzyshell.get_metadata('ann_command_count', '0'))
    current_count = fuzzyshell.get_indexed_count()
    poorly_clustered = int(fuzzyshell.get_metadata('poorly_clustered_commands', '0'))
    
    if ann_command_count == 0:
        return None
    
    growth = current_count - ann_command_count
    growth_pct = (growth / ann_command_count) * 100 if ann_command_count > 0 else 0
    
    if growth_pct > 15 or growth > 200 or poorly_clustered > 50:
        return (
            f"💡 ANN index rebuild recommended\n"
            f"   {growth} new commands ({growth_pct:.1f}% growth)\n"
            f"   Run: fuzzyshell --rebuild-ann"
        )
    
    return None