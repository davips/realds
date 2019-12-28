stack build && stack exec -- tricoll +RTS -M400m &> output.log
grep -e "^\[ " -e "^, " -e "^\]" output.log &> output.txt
rm output.log
