add_library(ALPR_Core STATIC
    
)

target_link_libraries(ALPR_Core
    PUBLIC
       ALPR_Headers
        ${OpenCV_LIBS}
)
