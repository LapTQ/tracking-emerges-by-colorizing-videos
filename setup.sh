#!/bin/bash

# Download directories vars
datasets_dir="data"
root_dl="${datasets_dir}/k700-2020"
root_dl_targz="${datasets_dir}/k700-2020_targz"
max_train_tar=100
max_val_tar=10
max_test_tar=10

# Make root directories
[ ! -d $root_dl ] && mkdir $datasets_dir
[ ! -d $root_dl ] && mkdir $root_dl
[ ! -d $root_dl_targz ] && mkdir $root_dl_targz

# Download train tars, will resume
curr_dl=${root_dl_targz}/train
[ ! -d $curr_dl ] && mkdir -p $curr_dl
# wget -c -i https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt -P $curr_dl
for((count=1; count <= $max_train_tar; count++)); do
	name=$(printf "%03d" "$count") && link="https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_$name.tar.gz" && echo "Downloading $link" && wget -c $link -P $curr_dl && echo "Downloaded $count/$max_train_tar"
done

# Download validation tars, will resume
curr_dl=${root_dl_targz}/val
[ ! -d $curr_dl ] && mkdir -p $curr_dl
# wget -c -i https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt -P $curr_dl
for((count=1; count <= $max_val_tar; count++)); do
	name=$(printf "%03d" "$count") && link="https://s3.amazonaws.com/kinetics/700_2020/val/k700_val_$name.tar.gz" && echo "Downloading $link" && wget -c $link -P $curr_dl && echo "Downloaded $count/$max_train_tar"
done

# Download test tars, will resume
curr_dl=${root_dl_targz}/test
[ ! -d $curr_dl ] && mkdir -p $curr_dl
# wget -c -i https://s3.amazonaws.com/kinetics/700_2020/test/k700_2020_test_path.txt -P $curr_dl
for((count=1; count <= $max_val_tar; count++)); do
	name=$(printf "%03d" "$count") && link="https://s3.amazonaws.com/kinetics/700_2020/test/k700_test_$name.tar.gz" && echo "Downloading $link" && wget -c $link -P $curr_dl && echo "Downloaded $count/$max_train_tar"
done

# Download k700-2020 annotations targz file from deep mind
curr_dl=${root_dl_targz}/annotations/deepmind 
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020.tar.gz -P $curr_dl

# Download k700-2020 annotations targz file from deep mind
curr_dl=${root_dl_targz}/annotations/deepmind_top-up
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020_delta.tar.gz -P $curr_dl

# Download AVA Kinetics
curr_dl=${root_dl_targz}/annotations/AVA-Kinetics
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c https://s3.amazonaws.com/kinetics/700_2020/annotations/ava_kinetics_v1_0.tar.gz -P $curr_dl
wget -c https://s3.amazonaws.com/kinetics/700_2020/annotations/countix.tar.gz -P $curr_dl

# Download annotations csv files
curr_dl=${root_dl}/annotations
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv -P $curr_dl
wget -c https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv -P $curr_dl
wget -c https://s3.amazonaws.com/kinetics/700_2020/annotations/test.csv -P $curr_dl

# Download readme
wget -c http://s3.amazonaws.com/kinetics/700_2020/K700_2020_readme.txt -P $root_dl

# Downloads complete
echo -e "\nDownloads complete!"


# Extract train and remove tar
curr_dl=$root_dl_targz/train
curr_extract=$root_dl/train
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract && rm $curr_dl/$f
done

# Extract validation and remove tar
curr_dl=$root_dl_targz/val
curr_extract=$root_dl/val
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract && rm $curr_dl/$f
done

# Extract test and remove tar
curr_dl=$root_dl_targz/test
curr_extract=$root_dl/test
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract && rm $curr_dl/$f
done

# Extract deep mind annotations and remove tar
curr_dl=$root_dl_targz/annotations/deepmind
curr_extract=$root_dl/annotations/deepmind
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract && rm $curr_dl/$f
done

# Extract deep mind top-up annotations and remove tar
curr_dl=$root_dl_targz/annotations/deepmind_top-up
curr_extract=$root_dl/annotations/deepmind_top-up
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract && rm $curr_dl/$f
done

# Extract deep mind top-up annotations and remove tar
curr_dl=$root_dl_targz/annotations/AVA-Kinetics
curr_extract=$root_dl/annotations/AVA-Kinetics
[ ! -d $curr_extract ] && mkdir -p $curr_extract
tar_list=$(ls $curr_dl)
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting $curr_dl/$f to $curr_extract && tar zxf $curr_dl/$f -C $curr_extract && rm $curr_dl/$f
done

# Extraction complete
echo -e "\nExtractions complete!"