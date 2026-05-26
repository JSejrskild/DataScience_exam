# This Python script transfers all participants VF files to /data and renames them to ParticipantID_filename

import os
import shutil
from glob import glob

source_base = "/work/verbal_fluency/VF_responses"
target_dir = "/work/verbal_fluency/data"

os.makedirs(target_dir, exist_ok=True)

for participant in os.listdir(source_base):
    if not participant.startswith("00"):
        continue

    participant_path = os.path.join(source_base, participant)
    response_path = os.path.join(participant_path, "responses")

    # Case 1: files inside "responses" folder
    if os.path.isdir(response_path):
        files_to_process = [
            os.path.join(response_path, f)
            for f in os.listdir(response_path)
            if os.path.isfile(os.path.join(response_path, f))
        ]

    # Case 2: files directly in participant folder
    else:
        files_to_process = [
            os.path.join(participant_path, f)
            for f in os.listdir(participant_path)
            if os.path.isfile(os.path.join(participant_path, f))
        ]

    # Copy + rename
    for source_file in files_to_process:
        filename = os.path.basename(source_file)
        new_filename = f"{participant}_{filename}"
        target_file = os.path.join(target_dir, new_filename)

        shutil.copy2(source_file, target_file)

    print(f"Processed {participant} ({len(files_to_process)} files)")

print("All done.")