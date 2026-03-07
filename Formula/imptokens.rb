# Homebrew formula for imptokens.
#
# This formula lives in the main repo. Tap it with:
#   brew tap nimhar/imptokens https://github.com/nimhar/imptokens
#   brew install imptokens
#
# After each GitHub Release, update `version` and the sha256 hashes:
#   curl -fsSL <release_url> | shasum -a 256

class Imptokens < Formula
  desc "Cut LLM token costs 30-60% with local semantic compression"
  homepage "https://github.com/nimhar/imptokens"
  license "MIT"
  version "0.1.0"

  on_macos do
    on_arm do
      url "https://github.com/nimhar/imptokens/releases/download/v#{version}/imptokens-Darwin-arm64"
      sha256 "369adcd4c4d52382df8284a145de015f28a40fd300a3b34a0b4da7a5e2924946"
    end

    on_intel do
      url "https://github.com/nimhar/imptokens/releases/download/v#{version}/imptokens-Darwin-x86_64"
      sha256 "5ef5657d7b024243402190227811e17f27593625029bbd6c714d973291d43737"
    end
  end

  def install
    # The downloaded file is a raw binary named after the URL basename.
    binary = Hardware::CPU.arm? ? "imptokens-Darwin-arm64" : "imptokens-Darwin-x86_64"
    bin.install binary => "imptokens"
  end

  def caveats
    <<~EOS
      Wire imptokens into Claude Code with one command:
        imptokens --setup-claude

      This writes a UserPromptSubmit hook to ~/.claude/settings.json so every
      large prompt is automatically compressed before it reaches the model.
      Restart Claude Code to activate.
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/imptokens --version")
  end
end
