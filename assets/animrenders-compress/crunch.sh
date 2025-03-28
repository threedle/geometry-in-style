for fname in "$@"
do
    outfname="$(basename "$fname")"
    echo "$fname -> $outfname"
    ffmpeg -y -i "$fname" -c:v libx264 -vf "scale=iw*0.84:-1" -crf 26 -preset veryslow -tune animation -movflags faststart "$outfname"
done
