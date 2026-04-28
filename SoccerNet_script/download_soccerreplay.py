#!/usr/bin/env python3
"""
Download SoccerReplay-1988 textual data (JSON) from HuggingFace.

Requires NDA-verified access to https://huggingface.co/datasets/Homie0609/SoccerReplay-1988
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print('huggingface_hub が必要です: pip install huggingface_hub')
    sys.exit(1)


REPO_ID = 'Homie0609/SoccerReplay-1988'


def _place_json(tmp_dir: str, out_json: str, split: str) -> None:
    """
    Extract classification_{split}.json from tmp_dir and save to out_json.
    
    Priority:
    1. Case A: classification_{split}.json exists directly
    2. Case B: Per-match JSON files (recursively search and aggregate)
    3. Case C: Pickle files (.pkl or .pickle)
    4. If none found: Display error and list tmp_dir contents
    """
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    
    # Case A: Search for classification_{split}.json file directly
    target_filename = f'classification_{split}.json'
    for root, dirs, files in os.walk(tmp_dir):
        if target_filename in files:
            src_file = os.path.join(root, target_filename)
            shutil.copy(src_file, out_json)
            return
    
    # Case B: Aggregate per-match JSON files
    json_files = []
    for root, dirs, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith('.json'):
                json_files.append(os.path.join(root, f))
    
    if json_files:
        aggregated = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        aggregated.extend(data)
                    elif isinstance(data, dict):
                        aggregated.extend(data.values())
            except Exception as e:
                print(f'[warn] Failed to load {json_file}: {e}')
        
        if aggregated:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(aggregated, f, ensure_ascii=False, indent=2)
            return
    
    # Case C: Search for pickle files
    pkl_files = []
    for root, dirs, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith('.pkl') or f.endswith('.pickle'):
                pkl_files.append(os.path.join(root, f))
    
    if pkl_files:
        aggregated = []
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        aggregated.extend(data)
                    elif isinstance(data, dict):
                        aggregated.extend(data.values())
            except Exception as e:
                print(f'[warn] Failed to load {pkl_file}: {e}')
        
        if aggregated:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(aggregated, f, ensure_ascii=False, indent=2)
            return
    
    # Case D: Not found
    print(f'[error] classification_{split}.json, JSON files, or pickle files not found in {tmp_dir}')
    print('[info] Contents of tmp_dir:')
    for root, dirs, files in os.walk(tmp_dir):
        level = root.replace(tmp_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download SoccerReplay-1988 dataset from HuggingFace'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace token (default: use HF_TOKEN environment variable)'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='train_data/json/SoccerReplay-1988',
        help='Output directory for extracted JSON files'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='train_data/hf_cache/SoccerReplay-1988',
        help='Cache directory for downloaded ZIP files'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid', 'test'],
        help='Splits to download (default: train valid test)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files'
    )
    
    args = parser.parse_args()
    
    # Resolve token
    token = args.token or os.getenv('HF_TOKEN')
    if not token:
        print('HuggingFace token が必要です。--token または HF_TOKEN 環境変数で指定してください。')
        sys.exit(1)
    
    # Download and extract each split
    split_stats = {}
    
    for split in args.splits:
        zip_filename = f'{split}.zip'
        out_json = os.path.join(args.out_dir, f'classification_{split}.json')
        
        if os.path.exists(out_json) and not args.force:
            print(f'[skip] {out_json} already exists')
            try:
                with open(out_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    split_stats[split] = len(data)
            except:
                pass
            continue
        
        try:
            print(f'Downloading {zip_filename} ...')
            zip_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=zip_filename,
                repo_type='dataset',
                token=token,
                cache_dir=args.cache_dir,
            )
            
            tmp_dir = os.path.join(args.cache_dir, f'{split}_extracted')
            os.makedirs(tmp_dir, exist_ok=True)
            
            print(f'Extracting {zip_filename} ...')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmp_dir)
            
            _place_json(tmp_dir, out_json, split)
            
            # Count items in output JSON
            with open(out_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                split_stats[split] = len(data)
            
            print(f'Saved: {out_json}')
        
        except Exception as e:
            error_msg = str(e)
            if 'GatedRepoError' in type(e).__name__ or '401' in error_msg or '403' in error_msg:
                print('アクセス権がありません。HuggingFace でデータセットの NDA に同意してください: '
                      'https://huggingface.co/datasets/Homie0609/SoccerReplay-1988')
                sys.exit(1)
            else:
                print(f'[error] Failed to download {zip_filename}: {e}')
                sys.exit(1)
    
    # Print summary
    print('\n' + '=' * 70)
    print('Download completed!')
    print('=' * 70)
    
    for split in args.splits:
        if split in split_stats:
            print(f'{split:8s}: {split_stats[split]:6d} items')
    
    # Print sample video path
    for split in args.splits:
        out_json = os.path.join(args.out_dir, f'classification_{split}.json')
        if os.path.exists(out_json):
            try:
                with open(out_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data and 'video' in data[0]:
                        print(f'\nSample video path ({split}): {data[0]["video"]}')
                        break
            except:
                pass
    
    print('\n注意: 動画ファイルは別途 NDA 申請後に提供されるリンクからダウンロードしてください。')


if __name__ == '__main__':
    main()
