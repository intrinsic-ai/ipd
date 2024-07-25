#! /bin/bash


ID=$1 # ID of the dataset
CAMERA=$2 # Camera name
FOLDER=$3 # Folder to store the dataset (no trailing slash)

################################################
# Function to download a file from a URL
################################################ 
download_file() {
    local URL="$1"
    local to_dir="$2"
    local file_name=$(basename "$URL")

    #check if to_dir folder exists
    if [ ! -d "$to_dir" ]; then
        mkdir "$to_dir"
    fi
    
    # Check if the file already exists in the target directory
    if [[ ! -f "$to_dir/$file_name" ]]; then
        echo "Downloading $URL"
        wget -O "$to_dir/$file_name" $URL
        if [ $? -ne 0 ]; then 
            echo "Could not download: $to_dir/$file_name"
            echo "URL: $URL"
            rm "$to_dir/$file_name"
            exit 1
        fi
    else
        echo "$to_dir/$file_name already exists."
    fi
}

################################################
# Download and extract dataset into ./datasets folder
################################################

DATASET=${CAMERA}-${ID}
URL=https://storage.googleapis.com/akasha-public/industrial_plenoptic_dataset/${DATASET}.zip

download_file "$URL" "$FOLDER"
if [ $? -ne 0 ]; then 
    echo "Could not download: $DATASET"
    exit 1
fi

unzip ${FOLDER}/${DATASET}.zip -d ${FOLDER}/

echo "Downloaded dataset: $DATASET"


################################################
# Download models into ./datasets/models folder
################################################
echo "Now...downloading models"

cad_files=(
    "corner_bracket.stl"
    "corner_bracket.stl"
    "corner_bracket0.stl"
    "corner_bracket1.stl"
    "corner_bracket2.stl"
    "corner_bracket3.stl"
    "corner_bracket4.stl"
    "corner_bracket6.stl"
    "gear1.stl"
    "gear2.stl"
    "handrail_bracket.stl"
    "hex_manifold.stl"
    "l_bracket.stl"
    "oblong_float.stl"
    "pegboard_basket.stl"
    "pipe_fitting_unthreaded.stl"
    "single_pinch_clamp.stl"
    "square_bracket.stl"
    "t_bracket.stl"
    "u_bolt.stl"
    "wraparound_bracket.stl"
)
CAD_MODELS_DIR="${FOLDER}/models"
for cad_file in "${cad_files[@]}"; do
  URL=https://storage.googleapis.com/akasha-public/industrial_plenoptic_dataset/cad_models/${cad_file}
  download_file "$URL" "$CAD_MODELS_DIR"
done
    