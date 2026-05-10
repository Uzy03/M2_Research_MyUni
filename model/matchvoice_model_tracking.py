import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import torch.nn.functional as F
import einops
from model.matchvoice_model_all_blocks import matchvoice_model_all_blocks, LayerNorm
from tracking.encoder import TrackingEncoder


class matchvoice_model_tracking(matchvoice_model_all_blocks):
    """
    matchvoice_model_all_blocks を継承し、visual_encoder だけ TrackingEncoder に差し替えるラッパー。
    LLaMA・Q-Former・LoRA は全て親クラスをそのまま流用する。
    
    入力形式:
        samples['tracking']: (B, T, N, F) - トラッキングテンソル
        samples['mask']: (B, T, N) - マスク (オプション)
    """
    
    def __init__(self, num_players=23, in_features=5, d_model=256, **kwargs):
        """
        Args:
            num_players (int): 選手数（デフォルト: 23）
            in_features (int): 入力特徴次元（デフォルト: 5）
            d_model (int): トランスフォーマー内部次元（デフォルト: 256）
            **kwargs: 親クラスに渡すその他のパラメータ
        """
        # load_checkpoint が渡されていたら False に強制
        if 'load_checkpoint' in kwargs:
            kwargs['load_checkpoint'] = False
        
        # visual_encoder を初期化させないために visual_encoder_checkpoint を設定
        kwargs.setdefault('visual_encoder_checkpoint', 'NONE')
        
        # 親クラスの __init__ を呼ぶ
        super().__init__(**kwargs)
        
        # Alignment loss 用の投影層を追加
        llm_hidden = self.llama_model.config.hidden_size
        self.align_proj = nn.Linear(llm_hidden, llm_hidden)
        self.slot_proj = nn.ModuleList([
            nn.Linear(llm_hidden, llm_hidden) for _ in range(3)
        ])
        
        # visual_encoder を TrackingEncoder で上書き
        out_features = kwargs.get('num_features', 768)
        self.visual_encoder = TrackingEncoder(
            num_players=num_players,
            in_features=in_features,
            d_model=d_model,
            out_features=out_features,
        )
    
    def forward(self, samples, validating=False, lambda_align=0.0, lambda_slot=0.0):
        """
        Forward pass: TrackingEncoder で処理したトラッキングデータを親クラスの
        Q-Former と LLaMA に渡す。
        
        Args:
            samples (dict): 入力サンプル
                - 'tracking': (B, T, N, F) トラッキングテンソル
                - 'mask': (B, T, N) マスク（オプション）
                - 'labels': (B, seq_len) ターゲットラベル
                - 'attention_mask': (B, seq_len) 注意マスク
                - 'input_ids': (B, seq_len) 入力トークンID
                - 'caption_text': (B,) キャプションテキスト
                - 'video_path': (B,) ビデオパス
            validating (bool): 検証モード（デフォルト: False）
        
        Returns:
            loss または (generated_text, caption_text, video_path)
        """
        # TrackingEncoder で追跡データを処理
        tracking_tensor = samples['tracking']  # (B, T, N, F)
        mask_tensor = samples.get('mask')      # (B, T, N) または None
        
        # TrackingEncoder の出力 (B, T, 768)
        video_features = self.visual_encoder(tracking_tensor, mask_tensor)
        
        # ここから親クラスの forward ロジックをほぼそのままコピーして使用
        targets = samples['labels']
        atts_llama = samples['attention_mask']
        inputs_ids = samples['input_ids']
        caption_text = samples['caption_text']
        video_path = samples['video_path']
        
        # video_features は既に TrackingEncoder から出力されているので、
        # 親クラスのように visual_encoder を呼ばない
        
        batch_size = None
        time_length = None
        try:
            batch_size, time_length, _ = video_features.size()
        except:
            batch_size, time_length, _, _ = video_features.size()
        
        if len(video_features.size()) != 4:
            video_features = video_features.unsqueeze(-2)
        video_features = self.ln_vision(video_features)
        video_features = einops.rearrange(video_features, 'b t n f -> (b t) n f', b=batch_size, t=time_length)
        
        if self.need_temporal == "yes":
            position_ids = torch.arange(time_length, dtype=torch.long, device=video_features.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(video_features, '(b t) n f -> b t n f', b=batch_size, t=time_length)
        
        if self.need_temporal == "yes":
            frame_hidden_state = frame_position_embeddings + frame_hidden_state
        
        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size, t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(frame_hidden_state)
        if self.qformer_heads > 1:
            head_outputs = []
            for h in range(self.qformer_heads):
                q_h = self.video_query_tokens[h].unsqueeze(0).expand(batch_size, -1, -1).to(frame_hidden_state.device)
                out_h = self.video_Qformer.bert(
                    query_embeds=q_h,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )
                head_outputs.append(out_h.last_hidden_state)
            video_hidden = torch.cat(head_outputs, dim=1)
        else:
            video_query_tokens = self.video_query_tokens.expand(batch_size, -1, -1).to(frame_hidden_state.device)
            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )
            video_hidden = video_query_output.last_hidden_state
        
        inputs_llama = self.llama_proj(video_hidden)

        slot_loss = None
        if lambda_slot > 0.0 and not validating and not self.inference:
            slot_labels = samples.get('slot_labels')
            if slot_labels is not None:
                if self.open_llm_decoder:
                    embed_fn = self.llama_model.base_model.model.model.embed_tokens
                else:
                    embed_fn = self.llama_model.model.embed_tokens
                tokens_per_slot = inputs_llama.shape[1] // 4
                slot_loss = torch.tensor(0.0, device=inputs_llama.device, dtype=torch.float)
                for s in range(3):
                    slot_mean = inputs_llama[:, s*tokens_per_slot:(s+1)*tokens_per_slot, :].float().mean(dim=1)
                    slot_projected = self.slot_proj[s](slot_mean.to(self.slot_proj[s].weight.dtype))
                    gt_embs = []
                    for row in slot_labels:
                        text = row[s] if row[s] else ''
                        if not text:
                            continue
                        ids = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to(inputs_llama.device)
                        with torch.no_grad():
                            emb = embed_fn(ids).float().mean(dim=1)
                        gt_embs.append(emb)
                    if gt_embs:
                        gt_tensor = torch.cat(gt_embs, dim=0)
                        slot_loss = slot_loss + (1.0 - F.cosine_similarity(slot_projected.float(), gt_tensor, dim=-1)).mean()
        
        # Alignment loss（学習時のみ、lambda_align > 0 の場合）
        align_loss = None
        if lambda_align > 0.0 and not validating and not self.inference:
            if self.open_llm_decoder:
                embed_fn = self.llama_model.base_model.model.model.embed_tokens
            else:
                embed_fn = self.llama_model.model.embed_tokens

            # h_prefix: (B, 32, H) → mean → (B, H)
            h_mean = inputs_llama.float().mean(dim=1)
            h_proj = self.align_proj(h_mean.to(self.align_proj.weight.dtype))

            # GT caption_text をトークナイズして埋め込み → mean
            gt_embeds_list = []
            for text in caption_text:
                ids = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to(inputs_llama.device)
                with torch.no_grad():
                    emb = embed_fn(ids).float().mean(dim=1)  # (1, H)
                gt_embeds_list.append(emb)
            gt_embeds = torch.cat(gt_embeds_list, dim=0)  # (B, H)

            align_loss = (1.0 - F.cosine_similarity(h_proj.float(), gt_embeds, dim=-1)).mean()
        
        if self.inference:
            return self.generate_text(inputs_llama)
        
        if validating:
            temp_res_text = self.generate_text(inputs_llama)
            return temp_res_text, caption_text, video_path
        
        n_vis_tokens = inputs_llama.shape[1]
        visual_label = torch.full((batch_size, n_vis_tokens), -100, dtype=targets.dtype).to(inputs_llama.device)
        concat_targets = torch.cat((visual_label, targets), dim=1).to(inputs_llama.device)
        temp_input_ids = inputs_ids.clone().to(inputs_llama.device)
        if self.open_llm_decoder == True:
            targets_embeds = self.llama_model.base_model.model.model.embed_tokens(temp_input_ids)
        else:
            targets_embeds = self.llama_model.model.embed_tokens(temp_input_ids)
        embedding_cat = torch.cat((inputs_llama, targets_embeds), dim=1)
        mask_prefix = torch.ones(batch_size, n_vis_tokens, dtype=atts_llama.dtype).to(inputs_llama.device)
        mask = torch.concat((mask_prefix, atts_llama), dim=1).to(inputs_llama.device)
        
        import io
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        with self.maybe_autocast(embedding_cat):
            outputs = self.llama_model(
                inputs_embeds=embedding_cat,
                attention_mask=mask,
                return_dict=True,
                labels=concat_targets,
            )
        sys.stdout = original_stdout
        loss = outputs.loss
        if align_loss is not None:
            loss = loss + lambda_align * align_loss
        if slot_loss is not None and lambda_slot > 0.0:
            loss = loss + lambda_slot * slot_loss
        return loss

    def forward_contrastive(self, samples):
        """TrackingEncoder(frozen) → Q-Former → llama_proj の出力を返す。対照学習用。"""
        tracking_tensor = samples['tracking']
        mask_tensor = samples.get('mask')

        with torch.no_grad():
            video_features = self.visual_encoder(tracking_tensor, mask_tensor)

        batch_size, time_length, _ = video_features.size()
        video_features = video_features.unsqueeze(-2)
        video_features = self.ln_vision(video_features)
        video_features = einops.rearrange(video_features, 'b t n f -> (b t) n f',
                                          b=batch_size, t=time_length)

        if self.need_temporal == "yes":
            position_ids = torch.arange(time_length, dtype=torch.long,
                                        device=video_features.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)

        frame_hidden_state = einops.rearrange(video_features, '(b t) n f -> b t n f',
                                              b=batch_size, t=time_length)
        if self.need_temporal == "yes":
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',
                                              b=batch_size, t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1],
                                dtype=torch.long).to(frame_hidden_state)

        video_query_tokens = self.video_query_tokens.expand(batch_size, -1, -1).to(
            frame_hidden_state.device)
        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )
        video_hidden = video_query_output.last_hidden_state
        inputs_llama = self.llama_proj(video_hidden)  # (B, 32, llm_hidden)
        return inputs_llama
