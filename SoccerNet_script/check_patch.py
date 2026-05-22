import json
import sys

json_path = sys.argv[1] if len(sys.argv) > 1 else 'soccerdata_clips/fps1_sec30_onball_step5s/clips.json'

d = json.load(open(json_path))
has   = sum(1 for e in d if 'action_sequence_frames' in e)
ok    = sum(1 for e in d if 'action_sequence_frames' in e
           and len(e['action_sequence_frames']) == len(e.get('action_sequence', [])))
bad   = has - ok
print(f'total={len(d)}, has_frames={has}, match={ok}, mismatch={bad}')

sample = next((e for e in d if e.get('action_sequence_frames')), None)
if sample:
    print('sample game_id       :', sample['game_id'])
    print('sample action_seq    :', sample.get('action_sequence'))
    print('sample action_frames :', sample.get('action_sequence_frames'))
else:
    print('WARNING: no entry with action_sequence_frames found')
