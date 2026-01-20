add_executable(ALPR_System
    main.cc
)

target_link_libraries(ALPR_System
    PRIVATE
        ALPR_Core
)

install(
    TARGETS ALPR_System
    RUNTIME DESTINATION bin
)
