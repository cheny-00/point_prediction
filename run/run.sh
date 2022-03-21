if [ "$#" -gt 0 ]
    then
        args="$*"
fi

python ../run_train.py $args
