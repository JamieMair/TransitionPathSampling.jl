module TPS

function do_nothing() nothing end

function get_init_fn(cache, config) end
function get_stopping_fn(cache, config) end
function get_step_fn(cache, config) end
function get_pre_init_hook(cache, config) do_nothing end
function get_post_init_hook(cache, config) do_nothing end
function get_pre_step_hook(cache, config) do_nothing end
function get_post_step_hook(cache, config) do_nothing end
function get_final_hook(cache, config) do_nothing end

function run!(cache, config)
    # Extract all the functions you need based on the configuration and cache
    init_fn! = get_init_fn(cache, config)
    stop_fn! = get_stopping_fn(cache, config)
    step_fn! = get_step_fn(cache, config)

    # Load the hook functions
    pre_init_hook! = get_pre_init_hook(cache, config)
    post_init_hook! = get_post_init_hook(cache, config)

    pre_step_hook! = get_pre_step_hook(cache, config)
    post_step_hook! = get_post_step_hook(cache, config)

    final_hook! = get_final_hook(cache, config)

    # Compile the inner loop, hopefully removing unneeded hooks
    function inner_loop!()
        pre_step_hook!()
        step_fn!()
        post_step_hook!()
    end

    pre_init_hook!()
    init_fn!()
    post_init_hook!()
    
    while !stop_fn!()
        inner_loop!()
    end

    final_hook!()

    nothing
end


end