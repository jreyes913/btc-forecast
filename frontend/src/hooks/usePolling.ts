import { useEffect, useRef, useState } from "react";
import { NotReadyError } from "../api";

/**
 * Runs `task` now, then again `intervalMs` after each run finishes (or `retryMs` after a
 * 503). A single setTimeout chain means retries never stack up, and everything stops on
 * unmount. Returns a status message to show while the backend isn't ready, or "" if OK.
 */
export function usePolling(
  task: (signal: AbortSignal) => Promise<void>,
  intervalMs = 15_000,
  retryMs = 5_000,
): string {
  const taskRef = useRef(task);
  taskRef.current = task;
  const [status, setStatus] = useState("");

  useEffect(() => {
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    const loop = async () => {
      let delay = intervalMs;
      try {
        await taskRef.current(ctrl.signal);
        setStatus("");
      } catch (err) {
        if (ctrl.signal.aborted) return;
        if (err instanceof NotReadyError) {
          setStatus(err.message);
          delay = retryMs;
        } else {
          console.error(err);
          setStatus("Unable to reach the backend API.");
        }
      }
      if (!ctrl.signal.aborted) timer = setTimeout(loop, delay);
    };

    loop();
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [intervalMs, retryMs]);

  return status;
}
