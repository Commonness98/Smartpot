"""스마트팜 에이전트 AI - Streamlit 대화형 웹 UI (목표④).

실행:
    python -m streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from smartfarm import registry, schema
from smartfarm.agent import llm_client, tuner
from smartfarm.agent.core import answer
from smartfarm.agent.planner import recommend_schedule
from smartfarm.agent.schema_mapper import propose_mapping
from smartfarm.data_pipeline import ingest_dataframe
from smartfarm.models.predict import recent_series
from smartfarm.models.train import train

st.set_page_config(page_title="스마트팜 에이전트 AI", page_icon="🌱", layout="wide")
registry.ensure_default()

st.title("🌱 스마트팜 에이전트 AI")
st.caption("온실 에너지(난방+전기) 수요 예측 및 작업 일정 권고를 위한 대화형 에이전트 — GrandICT 연구센터")


# 대화 스크롤 영역 높이(px). 이 안에서만 스크롤되므로 입력창 위치가 고정된다.
# 노트북 화면(약 800~900px)에서 입력창이 화면 밖으로 밀리지 않는 값.
CHAT_HEIGHT = 420


def _scroll_chat_to_bottom(nonce: int) -> None:
    """대화 영역을 항상 최신 메시지로 스크롤한다.

    st.container(height=)는 리런 후 스크롤이 맨 위로 돌아가 최신 답변이 보이지
    않는다. 영역 끝에 심어둔 앵커(#chat-end)를 부모 문서에서 찾아 스크롤한다.

    nonce: HTML이 매번 동일하면 Streamlit이 컴포넌트를 다시 렌더하지 않아
           스크립트가 재실행되지 않는다. 메시지 수를 넣어 매번 달라지게 한다.
    """
    components.html(
        f"<!-- render {nonce} -->" + """
<script>
  (function () {
    const doc = window.parent.document;
    // 렌더가 끝날 때까지 잠깐 재시도한다. scrollIntoView는 바깥 페이지가 스크롤될
    // 수 있어, 스크롤 컨테이너를 직접 찾아 scrollTop을 지정한다.
    let tries = 0;
    const tick = setInterval(function () {
      tries++;
      const end = doc.getElementById("chat-end");
      if (end) {
        let sc = end.parentElement;
        while (sc) {
          const oy = window.parent.getComputedStyle(sc).overflowY;
          if (sc.scrollHeight > sc.clientHeight + 5 &&
              (oy === "auto" || oy === "scroll")) {
            sc.scrollTop = sc.scrollHeight;
            clearInterval(tick);
            return;
          }
          sc = sc.parentElement;
        }
      }
      if (tries > 25) clearInterval(tick);
    }, 80);
  })();
</script>
""",
        height=0,
    )


def copy_button(text: str, key: str, label: str = "복사") -> None:
    """답변 텍스트를 클립보드에 복사하는 작은 버튼.

    components.html은 별도 iframe이라 Streamlit 테마를 물려받지 못하므로
    prefers-color-scheme으로 명암을 맞춘다. 샌드박스에서 navigator.clipboard가
    막히는 경우가 있어 execCommand 폴백을 둔다.
    """
    payload = json.dumps(text)
    components.html(
        f"""
<style>
  :root {{ color-scheme: light dark; }}
  .cp {{
    font: 500 12px/1.2 -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 10px; border-radius: 7px; cursor: pointer;
    background: transparent; color: #7a7f87;
    border: 1px solid rgba(128,128,128,.35);
    transition: background .12s, color .12s;
  }}
  .cp:hover {{ background: rgba(128,128,128,.14); color: #b9bec6; }}
  .cp.done {{ color: #2e9e5b; border-color: #2e9e5b66; }}
</style>
<button class="cp" id="{key}">
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <rect x="9" y="9" width="12" height="12" rx="2"/>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
  </svg><span>{label}</span>
</button>
<script>
  (function () {{
    const b = document.getElementById({json.dumps(key)});
    const t = {payload};
    b.addEventListener("click", async function () {{
      try {{
        await navigator.clipboard.writeText(t);
      }} catch (e) {{
        const ta = document.createElement("textarea");
        ta.value = t; ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta); ta.select();
        try {{ document.execCommand("copy"); }} catch (e2) {{}}
        ta.remove();
      }}
      const s = b.querySelector("span");
      s.textContent = "복사됨";
      b.classList.add("done");
      setTimeout(function () {{
        s.textContent = {json.dumps(label)}; b.classList.remove("done");
      }}, 1400);
    }});
  }})();
</script>
""",
        height=34,
    )


def _render_facts(facts: dict | None, verified, key, show_metrics=True) -> None:
    """어시스턴트 응답에 딸린 예측 지표·근거 데이터를 그린다.

    show_metrics: 사용자가 수치(수요·요금 등)를 실제로 물어봤을 때만 지표 카드를
                  노출한다. "방제해도 될까?" 같은 권고 질문에는 붙이지 않는다.
    """
    if not facts:
        return
    # 과거 실측 조회(history) 응답에는 예측 지표가 없다
    if show_metrics and "forecast" in facts:
        fc, sched = facts["forecast"], facts["schedule"]
        conf = facts.get("confidence", {})
        meth = facts.get("method", {})
        unit = fc.get("energy_unit", "")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"내일({fc['horizon_days']}일 뒤) 예측",
                  fc["predicted_energy"], sched["level"])
        c2.metric("최근 평균(14일)", fc["recent_avg_energy"])
        c3.metric("예상 전기요금", f"{sched['estimated_cost']:,.0f}원")

        # 단위는 지표마다 반복하지 않고 아래에 한 번만 표기한다.
        bits = [f"단위 {unit}"]
        if conf.get("range"):
            bits.append(f"오차 감안 {conf['range'][0]}~{conf['range'][1]}")
        if meth.get("method_label"):
            bits.append(f"산출 {meth['method_label']}")
        bits.append(f"요금은 {sched.get('area_m2', '-')}m² 기준")
        st.caption(" · ".join(bits))

        if conf.get("level") == "낮음":
            st.warning(f"⚠️ 신뢰도 낮음 — {conf['reasons'][0]}")
        st.caption(sched["advice"])
    if verified is False:
        st.caption("ℹ️ 생성 답변에 근거에 없는 수치가 있어 검증된 문장으로 대체했습니다.")
    with st.expander("🔎 근거 데이터(추론·계획)"):
        st.json(facts)


def _active_model_path() -> str | None:
    a = registry.get_active()
    if a and Path(a["model_path"]).exists():
        return a["model_path"]
    return None


# --- 사이드바: 데이터셋 선택 + 상태 ---
with st.sidebar:
    st.header("데이터셋")
    datasets = registry.list_datasets()
    active = registry.get_active()
    if datasets:
        ids = [d["id"] for d in datasets]
        labels = {d["id"]: d["name"] for d in datasets}
        idx = ids.index(active["id"]) if active and active["id"] in ids else 0
        chosen = st.selectbox("활성 데이터셋", ids, index=idx,
                              format_func=lambda i: labels[i])
        if active and chosen != active["id"]:
            registry.set_active(chosen)
            st.rerun()
        cur = registry.get_dataset(chosen)
        if cur and cur.get("metrics", {}).get("best_rmse"):
            m = cur["metrics"]
            st.caption(f"모델 RMSE {m['best_rmse']} · R² {m.get('best_r2', '-')} "
                      f"· {cur.get('rows', 0):,}행")

    st.divider()
    st.header("시스템 상태")
    st.write("**LM(로컬):**", "✅ 온라인" if llm_client.available() else "⚠️ 오프라인(규칙 기반)")
    st.write("**예측 모델:**", "✅ 로드됨" if _active_model_path() else "❌ 미학습")

active = registry.get_active()
active_model = _active_model_path()

tab_chat, tab_dash, tab_data = st.tabs(["💬 질의응답", "📊 대시보드", "📁 데이터 학습"])

# --- 대화형 질의응답 ---
with tab_chat:
    st.caption("자연어로 물어보세요 — 예: *“내일 에너지 수요가 어느 정도일까?”*, "
               "*“5월 3일 얼마 썼어?”*, *“제일 많이 쓴 날은?”*")

    if not active_model:
        st.warning("활성 데이터셋에 학습된 모델이 없습니다. '📁 데이터 학습' 탭에서 학습하세요.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 대화는 고정 높이의 스크롤 영역 안에 쌓는다. 대화가 길어져도 입력창이
        # 화면 아래로 밀려나지 않아, 매번 스크롤해서 찾을 필요가 없다.
        # (모든 메시지는 session_state에서만 렌더한다. 이전 구현은 응답을 입력창
        #  아래에 직접 그려 새 대화가 입력창 밑에 남고 순서가 뒤엉켰다.)
        box = st.container(height=CHAT_HEIGHT)
        with box:
            if not st.session_state.messages:
                st.caption("아직 대화가 없습니다. 아래에 질문을 입력해 보세요.")
            for i, m in enumerate(st.session_state.messages):
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if m["role"] == "assistant":
                        _render_facts(m.get("facts"), m.get("verified"), key=i,
                                      show_metrics=m.get("show_metrics", True))
                        copy_button(m["content"], key=f"msg{i}")
            # 자동 스크롤용 앵커(맨 아래)
            st.markdown('<div id="chat-end"></div>', unsafe_allow_html=True)
        if st.session_state.messages:
            _scroll_chat_to_bottom(len(st.session_state.messages))

        if q := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner("예측·계획 분석 중..."):
                # history: 직전까지의 대화(현재 질문 제외)를 넘겨 맥락 유지
                res = answer(q, model_path=active_model,
                             history=st.session_state.messages[:-1])
            st.session_state.messages.append({
                "role": "assistant", "content": res["text"],
                "facts": res.get("facts"), "verified": res.get("verified"),
                "show_metrics": res.get("show_metrics", True),
            })
            st.rerun()

# --- 대시보드 ---
with tab_dash:
    if not active_model:
        st.info("학습된 모델이 있어야 대시보드가 표시됩니다.")
    else:
        st.subheader("내일 에너지 수요 전망")
        sched = recommend_schedule(active_model)
        c1, c2, c3 = st.columns(3)
        c1.metric("예측 에너지", sched["predicted_energy"], sched["level"])
        c2.metric("최근 평균", sched["recent_avg_energy"])
        c3.metric("근사 비용(원)", sched["estimated_cost"])
        st.caption(sched["advice"])

        st.subheader("최근 일일 에너지 소비 추이")
        series = recent_series(active_model, days=60)
        trend = pd.DataFrame({"날짜": list(series.keys()), "에너지": list(series.values())})
        st.line_chart(trend.set_index("날짜"))

# --- 데이터 학습 (업로드 → 분류 → 확인 → 학습) ---
with tab_data:
    st.subheader("새 데이터 업로드 & 학습")
    st.markdown("CSV를 올리면 에이전트가 컬럼 역할을 자동 분류합니다. 표에서 확인·수정 후 학습하세요.")
    up = st.file_uploader("CSV 파일", type=["csv"])

    if up is not None:
        # 새 파일이면 원본과 분류 제안을 세션에 캐시(재실행 시 LLM 재호출 방지)
        if st.session_state.get("upload_name") != up.name:
            raw = pd.read_csv(up)
            st.session_state.upload_name = up.name
            st.session_state.upload_df = raw
            with st.spinner("컬럼 역할 자동 분류 중..."):
                st.session_state.upload_props = propose_mapping(raw)

        raw = st.session_state.upload_df
        st.write(f"미리보기 — {len(raw):,}행 × {raw.shape[1]}열")
        st.dataframe(raw.head(), use_container_width=True)

        prop_df = pd.DataFrame(st.session_state.upload_props)[
            ["column", "role", "confidence", "reason"]]
        prop_df.columns = ["컬럼", "역할", "신뢰도", "근거"]
        edited = st.data_editor(
            prop_df, use_container_width=True, hide_index=True,
            disabled=["컬럼", "신뢰도", "근거"],
            column_config={"역할": st.column_config.SelectboxColumn(
                "역할", options=schema.ALL_ROLES, required=True)})

        mapping = dict(zip(edited["컬럼"], edited["역할"]))
        roles = list(mapping.values())
        name = st.text_input("데이터셋 이름", value=up.name.rsplit(".", 1)[0])

        auto_tune = st.checkbox("🤖 자동 파라미터 튜닝", value=True,
                                help="hidden/layers/dropout/lr/seq_len 조합을 여러 개 탐색해 "
                                     "RMSE·R²가 가장 좋은 조합을 자동으로 채택합니다.")
        if auto_tune:
            c1, c2 = st.columns(2)
            n_trials = c1.number_input("탐색 횟수(trials)", 2, 20, 6)
            epochs = c2.number_input("트라이얼당 epoch", 3, 30, 8)
        else:
            epochs = st.number_input("학습 epoch", 3, 50, 12)

        # 유효성: timestamp/power_target 각 1개 이상
        problems = []
        if roles.count(schema.ROLE_TIMESTAMP) != 1:
            problems.append("timestamp 역할을 정확히 1개 지정하세요.")
        if roles.count(schema.ROLE_POWER) < 1:
            problems.append("power_target(전력 수요) 역할을 1개 이상 지정하세요.")
        for p in problems:
            st.warning(p)

        if st.button("✅ 수락하고 학습", type="primary", disabled=bool(problems)):
            ds_id, table, model_path = registry.new_ids(name)
            with st.status("학습 진행 중...", expanded=True) as status:
                st.write("1/3 표준 컬럼으로 변환·적재")
                rows = ingest_dataframe(raw, mapping, table)
                st.write(f"   {rows:,}행 적재 완료")

                if auto_tune:
                    st.write(f"2/3 자동 파라미터 튜닝 ({int(n_trials)}회 탐색)")

                    def _on_trial(i, n, combo, m):
                        st.write(f"   trial {i}/{n}: hidden={combo['hidden']} "
                                f"layers={combo['layers']} dropout={combo['dropout']} "
                                f"lr={combo['lr']} seq_len={combo['seq_len']} "
                                f"→ RMSE {m['best_rmse']} · R² {m['best_r2']}")

                    metrics = tuner.auto_train(table=table, model_path=model_path,
                                               mapping=mapping, epochs=int(epochs),
                                               n_trials=int(n_trials), progress_cb=_on_trial)
                    st.write(f"   최적 조합 {metrics['params']}")
                else:
                    st.write(f"2/3 LSTM 학습 ({epochs} epochs)")
                    metrics = train(table=table, model_path=model_path,
                                    mapping=mapping, epochs=int(epochs))

                st.write(f"   최적 RMSE {metrics['best_rmse']} · R² {metrics['best_r2']} "
                        f"(epoch {metrics['epoch']})")
                st.write("3/3 데이터셋 등록")
                registry.add_dataset(ds_id, name, table, mapping, model_path,
                                     metrics, rows, make_active=True)
                status.update(label=f"완료 · RMSE {metrics['best_rmse']} · R² {metrics['best_r2']}",
                             state="complete")
            st.session_state.pop("upload_name", None)
            st.success(f"'{name}' 학습 완료 & 활성화. 질의응답/대시보드에서 사용하세요.")
            st.rerun()

    st.divider()
    st.markdown("**등록된 데이터셋**")
    for d in registry.list_datasets():
        cols = st.columns([3, 2, 1])
        m = d.get("metrics", {})
        rmse = m.get("best_rmse", "-")
        r2 = m.get("best_r2", "-")
        cols[0].write(f"**{d['name']}**  \n`{d['id']}`")
        cols[1].write(f"RMSE {rmse} · R² {r2} · {d.get('rows', 0):,}행")
        if d["id"] != registry.DEFAULT_DATASET_ID:
            if cols[2].button("삭제", key=f"del_{d['id']}"):
                registry.delete_dataset(d["id"])
                st.rerun()
