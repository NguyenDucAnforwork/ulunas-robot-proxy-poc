const UlunasWasmModule = require('./ulunas_wasm.js');

UlunasWasmModule().then((Module) => {
    const HOP = 256;
    const reset = Module.cwrap('ulunas_wasm_reset', null, []);
    const processHop = Module.cwrap('ulunas_wasm_process_hop', null, ['number', 'number']);

    reset();

    const inPtr = Module._malloc(HOP * 4);
    const outPtr = Module._malloc(HOP * 4);

    const nHops = 40;
    const fullIn = new Float32Array(nHops * HOP);
    const fullOut = new Float32Array(nHops * HOP);
    // deterministic pseudo-random noise (avoid needing a seeded RNG lib)
    let seed = 12345;
    function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed / 0x7fffffff) * 2 - 1; }
    for (let i = 0; i < fullIn.length; i++) fullIn[i] = rnd() * 0.5;

    for (let h = 0; h < nHops; h++) {
        const chunk = fullIn.subarray(h * HOP, (h + 1) * HOP);
        Module.HEAPF32.set(chunk, inPtr / 4);
        processHop(inPtr, outPtr);
        const outChunk = new Float32Array(Module.HEAPF32.buffer, outPtr, HOP);
        fullOut.set(outChunk, h * HOP);
    }

    let maxErr = 0;
    const start = 4 * HOP;
    for (let i = start; i < nHops * HOP; i++) {
        const e = Math.abs(fullIn[i - HOP] - fullOut[i]);
        if (e > maxErr) maxErr = e;
    }
    console.log('WASM module loaded and ran successfully.');
    console.log('1-hop-delay-corrected identity max abs err:', maxErr);
    console.log(maxErr < 1e-4 ? 'PASS' : 'FAIL');

    Module._free(inPtr);
    Module._free(outPtr);
});
