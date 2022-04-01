# Research - Config

- These configs are added to the experiment search path such that any 
  files are found and read before that of the default experiment config.
  This means that if a file has the same name, it will overwrite the default file!
  The search path is overridden by setting the `DISENT_CONFIGS_PREPEND` environment variable.

- Additionally, we expose the research code by registering it with disent using the experiment
  plugin functionality. See `config/run_plugins`. The plugin will register each metric, framework
  or dataset with the `disent.registry`. Allowing easy use elsewhere through config entries.
