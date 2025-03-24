#!/bin/bash
set -e

# configure these variables before running!

# 1 less than the start ID of this run array
# Set this to how many runs you have done today
# (or a number that is at least as big as that.)
# Maybe 0, 10, 50, 100 etc nice round tens/hundreds numbers preferred)
BASEID=0

# source mesh dir for this run
MESHESDIR=victim-datasets/SMALsource-10-21-all

# directory where sds run results will be saved
RESULTSDIR=

# json file containing settings for all runs
BASEJSONFILE=

# directory of directories containing generated json files
PARENTJSONDUMPDIR=

# current month
MONTH=10

# current day
DAY=31

# nickname for this set of source meshes (nospaces)
DATASETNICKNAME=

# nickname for this run on this set of source meshes (nospaces)
RUNBATCHNICKNAME=


# END VARIABLES FOR CONFIG!
###########################################


OBJFILES=$(echo $MESHESDIR/*.obj | tr " " "\n" | grep -v preproc)
NRUNS=$(echo "$OBJFILES" | wc -l)
# we'll start from STARTID+1

STARTID=$((BASEID + 1))
ENDID=$((BASEID + NRUNS))

JSONDUMPDIR="$PARENTJSONDUMPDIR/$MONTH-$DAY.${STARTID}to$ENDID-$DATASETNICKNAME-$RUNBATCHNICKNAME/"

echo "json dump dir: $JSONDUMPDIR"
mkdir -p "$JSONDUMPDIR"
mkdir -p "$RESULTSDIR"

python scratch_generate_sds_run_configs.py \
    -j "$BASEJSONFILE" \
    -k "$BASEID" \
    -n "deform-csdv3-$MONTH-$DAY.{i}-$DATASETNICKNAME-{s}-$RUNBATCHNICKNAME" \
    -r "$RESULTSDIR" \
    -o "$JSONDUMPDIR" \
    -s nrmls \
    -i $OBJFILES