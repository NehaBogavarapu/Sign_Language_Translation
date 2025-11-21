import json

def parse_wlasl(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gloss_to_id = {}
    gloss_id = 0
    instances = []
    for entry in data:
        gloss = entry['gloss']
        if gloss not in gloss_to_id:
            gloss_to_id[gloss] = gloss_id
            gloss_id += 1
        gid = gloss_to_id[gloss]
        for inst in entry['instances']:
            vid = str(inst['video_id'])
            instances.append({
                'video_id': vid,
                'label': gid,
                'split': inst.get('split', 'train'),
                'bbox': inst.get('bbox', None)
            })
    return gloss_to_id, instances
