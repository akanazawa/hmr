# Invoke autocompletion
_complete_invoke() {
    local candidates
    candidates=`invoke --complete -- ${COMP_WORDS[*]}`
    COMPREPLY=( $(compgen -W "${candidates}" -- $2) )
}
# Tell shell builtin to use the above for completing our invocations.
complete -F _complete_invoke -o default invoke inv
