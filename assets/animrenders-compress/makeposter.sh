for fname in "$@"
do
    outfname="$(dirname "$fname")/poster-$(basename "$fname").jpg"
    echo "$fname -> $outfname"
    ffmpeg -y -i "$fname" -frames:v 1 -update 1 "$outfname"
done
