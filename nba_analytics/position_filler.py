"""
position_filler.py
-------------------

This module provides a complete pipeline to fill missing player positions in NBA stats DataFrames
using:
1. Name-matching
2. Height-based estimates
3. Manual corrections
4. Final validation
"""

import pandas as pd
import os


def fill_remaining_positions_with_height(df_player_stats, df_players):
    """
    Fill remaining players' positions using height-based estimation when no position data exists in df_players.
    Height-based estimation uses NBA positional height standards.
    """
    print("=== Final Position Fill Using Height Estimation ===")

    empty_mask = (df_player_stats['player_position'] == '')
    remaining_player_ids = df_player_stats[empty_mask]['player_id'].unique()
    print(f"Players needing height-based position estimation: {len(remaining_player_ids)}")

    remaining_players = df_players[df_players['id'].isin(remaining_player_ids)].copy()

    def parse_height_to_inches(height_str):
        if pd.isna(height_str) or str(height_str).strip() == '':
            return None
        try:
            height_str = str(height_str).strip()
            if '-' in height_str:
                feet, inches = height_str.split('-')
                return int(feet) * 12 + int(inches)
            return float(height_str)
        except (ValueError, AttributeError):
            return None

    def estimate_position_from_inches(total_inches):
        if total_inches is None:
            return 'G'
        if total_inches >= 82:
            return 'C'
        elif total_inches >= 79:
            return 'F'
        elif total_inches >= 77:
            return 'G-F'
        else:
            return 'G'

    position_estimates = {}
    print("\nHeight-based position estimates:")
    print("Player Name | Height | Inches | Estimated Position")
    print("-" * 70)

    for _, player in remaining_players.iterrows():
        player_id = player['id']
        name = f"{player['first_name']} {player['last_name']}"
        height = player['height']
        inches = parse_height_to_inches(height)
        estimated_pos = estimate_position_from_inches(inches)
        position_estimates[player_id] = estimated_pos
        print(f"{name:30} | {height:6} | {str(inches):6} | {estimated_pos}")

    updates_made = 0
    for idx in df_player_stats.loc[empty_mask].index:
        player_id = df_player_stats.loc[idx, 'player_id']
        if player_id in position_estimates:
            df_player_stats.loc[idx, 'player_position'] = position_estimates[player_id]
            updates_made += 1

    print(f"\nSuccessfully filled {updates_made} positions using height estimation")
    return df_player_stats, updates_made, position_estimates


def manual_position_corrections(df_player_stats, manual_overrides=None):
    """
    Apply manual position corrections for specific players.
    """
    if manual_overrides is None:
        manual_overrides = {
            2148: 'F', 2189: 'G', 2175: 'F', 1364: 'G',
            3547304: 'G', 9530711: 'F', 18678058: 'G',
            3091: 'G', 3092: 'F', 2208: 'G', 2073: 'G',
            5279: 'C', 2202: 'G'
        }

    print("=== Applying Manual Position Corrections ===")
    corrections_made = 0
    for player_id, correct_position in manual_overrides.items():
        player_mask = (df_player_stats['player_id'] == player_id)
        player_rows = df_player_stats[player_mask]
        if len(player_rows) > 0:
            player_name = f"{player_rows.iloc[0]['player_first_name']} {player_rows.iloc[0]['player_last_name']}"
            current_position = player_rows.iloc[0]['player_position']
            df_player_stats.loc[player_mask, 'player_position'] = correct_position
            corrections_made += len(player_rows)
            print(f"{player_name} (ID: {player_id}): '{current_position}' -> '{correct_position}' ({len(player_rows)} rows)")
    print(f"\nTotal corrections made: {corrections_made}")
    return df_player_stats, corrections_made


def validate_final_results(df_player_stats):
    """
    Validate and report on the final state of position filling.
    """
    total_rows = len(df_player_stats)
    empty_positions = (df_player_stats['player_position'] == '').sum()
    filled_positions = total_rows - empty_positions
    fill_rate = filled_positions / total_rows * 100

    print("\n=== Final Validation ===")
    print(f"Total rows: {total_rows:,}")
    print(f"Filled positions: {filled_positions:,} ({fill_rate:.2f}%)")
    print(f"Empty positions: {empty_positions:,}")

    print("\nPosition distribution:")
    print(df_player_stats['player_position'].value_counts(dropna=False))

    return {
        'total_rows': total_rows,
        'filled_positions': filled_positions,
        'empty_positions': empty_positions,
        'fill_rate': fill_rate
    }


def fill_by_name_matching(df_player_stats, df_players):
    """
    Fill missing positions using exact name matching between df_player_stats and df_players.
    """
    print("=== Name Matching Position Fill ===")

    valid_positions = df_players[df_players['position'] != ''].copy()
    valid_positions['full_name'] = (valid_positions['first_name'] + '|' +
                                     valid_positions['last_name']).str.upper().str.strip()

    name_position_map = dict(zip(valid_positions['full_name'], valid_positions['position']))
    empty_mask = (df_player_stats['player_position'] == '')
    updates = 0

    for idx in df_player_stats.loc[empty_mask].index:
        first_name = str(df_player_stats.loc[idx, 'player_first_name']).upper().strip()
        last_name = str(df_player_stats.loc[idx, 'player_last_name']).upper().strip()
        full_name = f"{first_name}|{last_name}"
        if full_name in name_position_map:
            df_player_stats.loc[idx, 'player_position'] = name_position_map[full_name]
            updates += 1

    print(f"Filled {updates} positions using name matching")
    return df_player_stats, updates


def complete_position_filling_pipeline(df_player_stats, df_players, use_manual_corrections=True):
    """
    Run the complete position-filling pipeline.
    """
    print("=== NBA POSITION FILLING PIPELINE ===")

    original_empty = (df_player_stats['player_position'] == '').sum()
    print(f"Starting with {original_empty:,} empty positions\n")

    # 1. Fill by name matching
    df_player_stats, name_updates = fill_by_name_matching(df_player_stats, df_players)

    # 2. Fill by height estimates
    df_player_stats, height_updates, estimates = fill_remaining_positions_with_height(df_player_stats, df_players)

    # 3. Fill by manual corrections
    manual_updates = 0
    if use_manual_corrections:
        df_player_stats, manual_updates = manual_position_corrections(df_player_stats)

    # 4. Validate results
    results = validate_final_results(df_player_stats)
    return df_player_stats, {
        'original_empty': original_empty,
        'name_fills': name_updates,
        'height_fills': height_updates,
        'manual_corrections': manual_updates,
        'final_empty': results['empty_positions'],
        'fill_rate': results['fill_rate'],
        'total_rows': results['total_rows']
    }

def load_parquet_to_dataframe(directory: str, file_name: str) -> pd.DataFrame:
    """
    Loads a Parquet file into a pandas DataFrame.

    Args:
        directory (str): The directory where the Parquet file is located.
        file_name (str): The name of the Parquet file.
    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    file_path = os.path.join(directory, file_name)
    df = pd.read_parquet(file_path)
    return df


if __name__ == "__main__":
    # Define file directory and paths
    #RAW_DATA_DIR = "../data/raw"
    #stats_pq_file = "player_game_stats_seasons_2021_2022_2023_2024_2025.parquet"
    #players_pq_file = "all_players_data_sdk.parquet"
    
    # Load player stats data
    df_player_stats = load_parquet_to_dataframe(RAW_DATA_DIR, stats_pq_file)
    print(df_player_stats.shape)
    
    # Load player data
    df_players = load_parquet_to_dataframe(RAW_DATA_DIR, players_pq_file)
    print(df_players.shape)
    print("Run complete_position_filling_pipeline with your DataFrames to fill positions!")
