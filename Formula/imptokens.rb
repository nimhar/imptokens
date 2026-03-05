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
      sha256 "aafdeaf0c56238da26e19e7f2ae40f4940b145b18020caa045d6b5c014b002ae"
    end

    on_intel do
      url "https://github.com/nimhar/imptokens/releases/download/v#{version}/imptokens-Darwin-x86_64"
      sha256 "e76be14adcf8c7212828ea42319188d18851db270a95c79cc6cb60ffa960b9e0"
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
