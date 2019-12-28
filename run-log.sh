stack build && stack exec -- tricoll +RTS -M400m | grep -e "^\[ " -e "^, " -e "^\]" &> output.log
