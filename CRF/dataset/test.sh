#!/bin/bash

for i in {301..400}
do
    crf_test -m crf_model ./result/一般项目/一般项目-$i.segment.txt > ./result/一般项目/一般项目-$i.crf.txt
    crf_test -m crf_model ./result/出院情况/出院情况-$i.segment.txt > ./result/出院情况/出院情况-$i.crf.txt
    crf_test -m crf_model ./result/诊疗经过/诊疗经过-$i.segment.txt > ./result/诊疗经过/诊疗经过-$i.crf.txt
    crf_test -m crf_model ./result/病史特点/病史特点-$i.segment.txt > ./result/病史特点/病史特点-$i.crf.txt
done