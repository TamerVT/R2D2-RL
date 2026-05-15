# This replaces the original libfranka function to bypass git errors
function(set_version_from_git VERSION_VAR TAG_VAR)
    set(${VERSION_VAR} "" PARENT_SCOPE)
    set(${TAG_VAR}   "" PARENT_SCOPE)
    message(STATUS "Patched libfranka to bypass git check")
endfunction()