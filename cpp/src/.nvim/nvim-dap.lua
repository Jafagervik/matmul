local dap = require("dap")

dap.adapters.codelldb = {
    type = "executable",
    command = "usr/bin/codelldb",
    name = "codelldb"
}

dap.configurations.cpp = {
    {
        name = "run",
        type = "codelldb",
        request = "launch",
        program = vim.fn.getcwd() .. "../../build/mandel",
        cwd = "${workspaceFolder}",
        stopOnEntry = false,
        args = { "./test.gflux" }, -- Probably text file
    },
}
